s# coding=utf-8

import torch
import torch.nn.functional as F
from torch import nn

import src.utils.init as my_init
from src.data.vocabulary import PAD, BOS, MOS, EOS, UNK
from src.decoding.utils import tile_batch, tensor_gather_helper
from src.modules.attention import GeneralAttention
from src.modules.embeddings import Embeddings
from src.modules.rnn import RNN
import torch.nn.utils.rnn as rnn_utils


class Encoder(nn.Module):
    def __init__(self,
                 n_words,
                 input_size,
                 hidden_size
                 ):
        """
        :param n_words:  词典的大小
        :param input_size:  嵌入词向量的大小
        :param hidden_size:  隐层的大小
        """
        super(Encoder, self).__init__()

        # Use PAD
        self.embeddings = Embeddings(num_embeddings=n_words,
                                     embedding_dim=input_size,
                                     dropout=0.0,
                                     add_position_embedding=False)

        self.gru = RNN(type="gru", batch_first=True, input_size=input_size, hidden_size=hidden_size,
                       bidirectional=True)

    def forward(self, x):
        """
        :param x: Input sequence.
            with shape [batch_size, seq_len]
        """

        # detach() x和x.detach()公用同一块地址，但是x.detach()不需要梯度
        # x_mask: [batch_size, seq_len] 由0和1组成，表示对应位置是否为PAD
        x_mask = x.detach().eq(PAD)

        # emb: [batch_size, seq_len, input_size]
        emb = self.embeddings(x)

        ctx, _ = self.gru(emb, x_mask)

        # ctx: [batch_size, src_seq_len, hid_dim*num_directions=context_size]
        # x_mask: [batch_size, seq_len]
        return ctx, x_mask


class Generator(nn.Module):

    def __init__(self, n_words, input_size, shared_weight=None, padding_idx=-1):

        super(Generator, self).__init__()

        self.n_words = n_words

        # 在调用的时候，这个hidden_size 输入的是 d_word_vec
        self.input_size = input_size
        self.padding_idx = padding_idx
        # 投影层，将hidden_size 投影到 n_words, 也就是将d_word_vec转变成n_words, 和嵌入层对应
        self.proj = nn.Linear(self.input_size, self.n_words, bias=False)

        if shared_weight is not None:
            self.proj.weight = shared_weight
        else:
            self._reset_parameters()

    def _reset_parameters(self):

        my_init.embedding_init(self.proj.weight)

    def _pad_2d(self, x):
        """
        不能预测 填充字符PAD
        """
        if self.padding_idx == -1:
            return x
        else:
            x_size = x.size()
            x_2d = x.view(-1, x.size(-1))

            mask = x_2d.new(1, x_2d.size(-1)).zero_()
            mask[0][self.padding_idx] = float('-inf')
            x_2d = x_2d + mask  # 这里是广播

            return x_2d.view(x_size)

    def _special_2d(self, x):
        """
        不能预测所有的特殊字符
        :param x:
        :return:
        """
        if self.padding_idx == -1:
            return x
        else:
            x_size = x.size()
            x_2d = x.view(-1, x.size(-1))

            mask = x_2d.new(1, x_2d.size(-1)).zero_()
            mask[0][PAD] = float('-inf')
            mask[0][EOS] = float('-inf')
            mask[0][MOS] = float('-inf')
            mask[0][BOS] = float('-inf')
            mask[0][UNK] = float('-inf')
            x_2d = x_2d + mask  # 这里是广播

            return x_2d.view(x_size)

    def forward(self, input, log_probs=True, special_mask=False):
        """
        input: [batch_size, tgt_seq_len, hidden_size]
        input == > Linear == > LogSoftmax
        """
        # 其实这里传进来的时emb_dim, 将emb_dim 转变成 vocab_size
        # [batch_size, tgr_seq_len, vocab_size]
        logits = self.proj(input)

        if special_mask:
            logits = self._special_2d(logits)
        else:
            logits = self._pad_2d(logits)

        # return: [batch_size, tgt_seq_len,  vocab_size]
        if log_probs:
            return torch.nn.functional.log_softmax(logits, dim=-1)
        else:
            return torch.nn.functional.softmax(logits, dim=-1)


class Decoder(nn.Module):

    def __init__(self, hidden_size, context_size, n_words, input_size, dropout=0.0, proj_share_weight=False,
                 bridge_type='mlp'):

        super().__init__()

        # 两个decoder使用一个embedding
        self.vocab_size = n_words
        self.bridge_type = bridge_type
        self.context_size = context_size
        self.hidden_size = hidden_size

        # 构造一个分类器，传入一个context的平均的隐层，然后经过generator生成
        self.middle_linear = nn.Linear(context_size, input_size)

        self.embeddings = Embeddings(num_embeddings=n_words, embedding_dim=input_size, dropout=dropout,
                                     add_position_embedding=False)

        # 注意力层
        self.enc_attn = GeneralAttention(query_size=hidden_size, value_size=context_size)
        self.dec_attn = GeneralAttention(query_size=hidden_size, value_size=hidden_size)
        self.out_attn = GeneralAttention(query_size=hidden_size, value_size=input_size)

        self.rnn_input = nn.Linear(hidden_size * 4, hidden_size)

        self.rnn_right = nn.GRU(batch_first=True, input_size=hidden_size,
                                hidden_size=hidden_size, dropout=dropout, bidirectional=False)
        self.rnn_left = nn.GRU(batch_first=True, input_size=hidden_size,
                               hidden_size=hidden_size, dropout=dropout, bidirectional=False)

        # 为了共享权重
        self.out = nn.Linear(in_features=hidden_size, out_features=input_size)

        if proj_share_weight is False:
            generator = Generator(n_words=n_words, input_size=input_size, padding_idx=PAD)
        else:
            generator = Generator(n_words=n_words, input_size=input_size, padding_idx=PAD,
                                  shared_weight=self.embeddings.embeddings.weight)
        self.generator = generator

        self._build_bridge()
        self._reset_parameters()

    def _reset_parameters(self):
        my_init.default_init(self.out.weight)
        my_init.default_init(self.rnn_input.weight)
        my_init.default_init(self.middle_linear.weight)
        for weight in self.rnn_left.parameters():
            my_init.rnn_init(weight)
        for weight in self.rnn_right.parameters():
            my_init.rnn_init(weight)

    # encoder到decoder的上下文
    def _build_bridge(self):

        self.linear_bridge = nn.Linear(in_features=self.context_size, out_features=self.hidden_size)
        my_init.default_init(self.linear_bridge.weight)

    def init_decoder(self, context, mask):
        """
        这里传入的时encoder的context和mask
        :param context: [batch_size, seq_len, context_size]
        :param mask: [batch_size, seq_len]
        :return:
        """
        no_pad_mask = 1.0 - mask.float()

        ctx_mean = (context * no_pad_mask.unsqueeze(2)).sum(1) / no_pad_mask.unsqueeze(2).sum(1)

        dec_init = F.tanh(self.linear_bridge(ctx_mean))

        enc_cache = self.enc_attn.compute_cache(context)
        # dec_init: [batch_size, hid_dim]
        # enc_cache: [batch_size, hidden_size, src_len]
        return dec_init, enc_cache

    def get_middle_word(self, trg, ctx, ctx_mask, inference=False):
        """
        根据encoder的最后一层，计算middle word的每个词的概率，选取出现在trg中概率最大的词语
        :param trg: [batch_size, trg_seq_len]
        :param ctx:
        :param ctx_mask:
        """
        # [batch_size, input_size]
        # no_pad_mask = 1.0 - ctx_mask.float()
        # ctx_mean = (ctx * no_pad_mask.unsqueeze(2)).sum(1) / no_pad_mask.unsqueeze(2).sum(1)
        # middle_linear_out = F.tanh(self.middle_linear(ctx_mean))
        #
        # middle_prob = self.generator(middle_linear_out, special_mask=True, log_probs=True)
        #
        # batch_size = ctx.shape[0]
        # range_ = (torch.arange(0, batch_size) * self.vocab_size).long().cuda().unsqueeze(1)
        # trg_index = trg + range_
        # select_middle_prob = torch.index_select(middle_prob.view(-1), -1, trg_index.view(-1)).view(batch_size, -1)
        #
        # # [batch_size]
        # max_middle_prob, max_index = torch.max(select_middle_prob, dim=-1)
        # range_ = (torch.arange(0, batch_size) * trg.shape[-1]).long().cuda()
        # _max_index = max_index + range_
        # middle_words = torch.index_select(trg.view(-1), -1, _max_index.view(-1)).view(batch_size)

        batch_size = ctx.shape[0]  # [EOS]

        middle_words = trg[:, 1]
        # max_index = torch.ones((batch_size, 1), dtype=torch)
        # [batch_size, emb_dim]
        embeddings = self.embeddings(middle_words)

        outputs = embeddings.new(batch_size, self.vocab_size).zero_().float()
        for i in range(batch_size):
            outputs[i][middle_words[i]] = 1
        if not inference:
            # 处理成left_sequences 和 right_sequence
            left_sequences = []
            right_sequences = []
            for i in range(batch_size):
                # 需要把左边的反转
                left_sequences.append(trg[i][0:1])
                right_sequences.append(trg[i][2:])

            # pad 填充
            left_trg = rnn_utils.pad_sequence(left_sequences, batch_first=True)
            right_trg = rnn_utils.pad_sequence(right_sequences, batch_first=True)
        else:
            left_trg = None
            right_trg = None

        sorted_trg = middle_words
        return embeddings, outputs, left_trg, right_trg, sorted_trg

    def inference(self, right_next, hiddens, embeddings, ctx, ctx_mask, enc_cache, trg_mask):
        if right_next:
            forward_hidden = hiddens[:, -1, :].contiguous()
            forward_emb = embeddings[:, -1, :]
            hidden, logits = self.one_step(forward_hidden, ctx, ctx_mask, embeddings, hiddens, forward_emb,
                                           enc_cache, trg_mask, right_next)

            hiddens = torch.cat((hiddens, hidden.unsqueeze(1)), dim=1)

        else:
            backward_hidden = hiddens[:, 0, :].contiguous()
            backward_emb = embeddings[:, 0, :]
            hidden, logits = self.one_step(backward_hidden, ctx, ctx_mask, embeddings, hiddens, backward_emb,
                                           enc_cache, trg_mask, right_next)

            hiddens = torch.cat((hidden.unsqueeze(1), hiddens), dim=1)

        return hiddens, logits

    # 单步执行这个函数
    def one_step(self, hidden, ctx, ctx_mask, embeddings, hiddens, emb, enc_cache=None, trg_mask=None,
                 right_next=False):
        """
        :param hidden:
        :param ctx:
        :param ctx_mask:
        :param embeddings:
        :param hiddens:
        :param emb: 当前decoder上一个输出层
        :param enc_cache:
        :param trg_mask:
        :param right_next:
        :return:
        """
        enc_context = self.enc_attn(query=hidden, value=ctx, cache=enc_cache, value_mask=ctx_mask)
        dec_context = self.dec_attn(query=hidden, value=hiddens, value_mask=trg_mask)
        out_context = self.out_attn(query=hidden, value=embeddings, value_mask=trg_mask)

        # input：[batch_size, hidden_size*4]
        rnn_input_input = torch.cat((enc_context, dec_context, out_context, emb), dim=-1)
        rnn_input_output = self.rnn_input(rnn_input_input)

        if right_next:
            rnn_output, hidden = self.rnn_right(rnn_input_output.unsqueeze(1), hidden.unsqueeze(0))
        else:
            rnn_output, hidden = self.rnn_left(rnn_input_output.unsqueeze(1), hidden.unsqueeze(0))

        hidden = hidden.squeeze(0)
        output = self.out(rnn_output.squeeze(1))
        logits = self.generator(output, log_probs=True, special_mask=False)
        return hidden, logits

    def forward(self, left_trg, right_trg, ctx, ctx_mask, embeddings, hiddens, outputs, sorted_trg, enc_cache, trg,
                trg_mask=None):
        forward_hidden = hiddens[:, 0, :]
        backward_hidden = hiddens[:, 0, :]
        forward_emb = embeddings[:, 0, :]
        backward_emb = embeddings[:, 0, :]

        left_seq_len = left_trg.shape[-1]
        right_seq_len = right_trg.shape[-1]

        seq_len = max(left_seq_len, right_seq_len)
        for t in range(seq_len):
            # 先往右翻译，后向左翻译
            if t < right_seq_len:
                forward_hidden, logits = self.one_step(forward_hidden, ctx, ctx_mask, embeddings, hiddens, forward_emb,
                                                       enc_cache, trg_mask, right_next=True)
                hiddens = torch.cat((hiddens, forward_hidden.unsqueeze(1)), dim=1)

                # 相当于使用teacher_forcing
                decoder_input = right_trg[:, t]
                forward_emb = self.embeddings(decoder_input)
                embeddings = torch.cat((embeddings, forward_emb.unsqueeze(1)), dim=1)

                _trg_mask = decoder_input.detach().eq(PAD) + decoder_input.detach().eq(EOS) + decoder_input.detach().eq(
                    BOS) + decoder_input.detach().eq(MOS)
                trg_mask = torch.cat((trg_mask, _trg_mask.unsqueeze(1)), dim=1)
                trg_mask.requires_grad = False

                outputs = torch.cat((outputs, logits.unsqueeze(1)), dim=1)
                sorted_trg = torch.cat((sorted_trg, decoder_input.unsqueeze(1)), dim=-1)

            if t < left_seq_len:
                backward_hidden, logits = self.one_step(backward_hidden, ctx, ctx_mask, embeddings, hiddens,
                                                        backward_emb,
                                                        enc_cache, trg_mask, right_next=False)

                hiddens = torch.cat((backward_hidden.unsqueeze(1), hiddens), dim=1)

                decoder_input = left_trg[:, t]
                backward_emb = self.embeddings(decoder_input)
                embeddings = torch.cat((backward_emb.unsqueeze(1), embeddings), dim=1)

                _trg_mask = decoder_input.detach().eq(PAD) + decoder_input.detach().eq(EOS) + decoder_input.detach().eq(
                    BOS) + decoder_input.detach().eq(MOS)
                trg_mask = torch.cat((_trg_mask.unsqueeze(1), trg_mask), dim=1)
                trg_mask.requires_grad = False

                outputs = torch.cat((outputs, logits.unsqueeze(1)), dim=1)
                sorted_trg = torch.cat((sorted_trg, decoder_input.unsqueeze(1)), dim=-1)
        # 所以这里的outputs并不是严格的做一个有一个的
        return outputs, sorted_trg


class Middle_L2R(nn.Module):
    """
    模型整体架构
    """

    # 处理1层
    def __init__(self, n_src_vocab, d_word_vec, d_model, n_tgt_vocab, proj_share_weight=False,
                 dropout=0.1, bridge_type='mlp', n_layers=1, **kwargs):
        super().__init__()

        self.input_size = d_word_vec
        self.hidden_size = d_model

        self.encoder = Encoder(n_src_vocab, d_word_vec, d_model)

        self.decoder = Decoder(d_model, d_model * 2, n_tgt_vocab, d_word_vec, dropout=dropout,
                               proj_share_weight=proj_share_weight, bridge_type=bridge_type)

        # self.classifier = Classifier(enc_hid_dim, decoder_vocab_size)

    def encode(self, src_seq):
        # ctx: [batch_size, seq_len, hid_dim*2=context_dim]
        # ctx_mask: [batch_size, seq_len]
        ctx, ctx_mask = self.encoder(src_seq)

        return {"ctx": ctx, "ctx_mask": ctx_mask}

    def init_decoder(self, enc_outputs, expand_size=1):
        """
        beam_search时调用, 多beam相当于增大了batch_size
        :param enc_outputs:
        :param expands_size:
        :return:
        """
        ctx = enc_outputs['ctx']
        ctx_mask = enc_outputs['ctx_mask']
        trg = enc_outputs['trg']

        dec_init, enc_cache = self.decoder.init_decoder(ctx, ctx_mask)
        embeddings, outputs, _, _, middle_words = self.decoder.get_middle_word(trg, ctx, ctx_mask, inference=True)

        trg_mask = torch.zeros(ctx.shape[0], dtype=torch.uint8).cuda()
        trg_mask.requires_grad = False

        if expand_size > 1:
            # 同一个batch的几个beam_size会靠着
            ctx = tile_batch(ctx, expand_size)
            ctx_mask = tile_batch(ctx_mask, expand_size)
            dec_init = tile_batch(dec_init, expand_size)
            outputs = tile_batch(outputs, expand_size)
            trg_mask = tile_batch(trg_mask, expand_size)
            embeddings = tile_batch(embeddings, expand_size)
            enc_cache = tile_batch(enc_cache, expand_size)

        # [batch_size, trg_seq_len, dec_hid_dim]
        hiddens = dec_init.unsqueeze(1)

        # [batch_size, trg_seq_len, vocab_size]
        outputs = outputs.unsqueeze(1)
        # [batch_size, trg_seq_len, vocab_size]
        embeddings = embeddings.unsqueeze(1)

        left_stop = torch.zeros(ctx.shape[0], dtype=torch.uint8).cuda()
        right_stop = torch.zeros(ctx.shape[0], dtype=torch.uint8).cuda()

        result = {
            "ctx": ctx,
            "ctx_mask": ctx_mask,
            "hiddens": hiddens,
            "embeddings": embeddings,
            "outputs": outputs,
            "trg_mask": trg_mask.unsqueeze(1),
            "right_next": True,
            "enc_cache": enc_cache,
            "middle_words": middle_words,
            "right_stop": right_stop,
            "left_stop": left_stop
        }

        return result

    def init_decoder_when_train(self, ctx, ctx_mask, trg):

        dec_init, enc_cache = self.decoder.init_decoder(ctx, ctx_mask)
        embeddings, outputs, left_trg, right_trg, sorted_trg = self.decoder.get_middle_word(trg, ctx, ctx_mask)

        hiddens = dec_init.unsqueeze(1)
        embeddings = embeddings.unsqueeze(1)
        outputs = outputs.unsqueeze(1)
        sorted_trg = sorted_trg.unsqueeze(1)

        trg_mask = torch.zeros(ctx.shape[0], dtype=torch.uint8).cuda().unsqueeze(1)
        trg_mask.requires_grad = False

        return embeddings, hiddens, outputs, trg_mask, enc_cache, left_trg, right_trg, sorted_trg

    def decode(self, dec_states):
        """
        trg_seq: 目前已经生成的decoder
        目前先只能单个操作把， 如果单个停止了，预测结果手动改成PAD？
        :param tgt_seq:
        :param dec_states:
        :return:
        """
        ctx = dec_states['ctx']
        ctx_mask = dec_states['ctx_mask']
        hiddens = dec_states['hiddens']
        embeddings = dec_states['embeddings']
        outputs = dec_states['outputs']
        trg_mask = dec_states['trg_mask']
        right_next = dec_states['right_next']
        enc_cache = dec_states['enc_cache']
        # embedding 使用beam_search的final結果
        hiddens, logits = self.decoder.inference(right_next, hiddens, embeddings, ctx, ctx_mask,
                                                 enc_cache, trg_mask)

        outputs = torch.cat((outputs, logits.unsqueeze(1)), dim=1)

        dec_states = {
            "ctx": ctx,                # 这个不用变
            "ctx_mask": ctx_mask,      # 这个不用变
            "hiddens": hiddens,        # 这个变了
            "embeddings": embeddings,  # 这个还没有改变
            "outputs": outputs,        # 这个变了
            "trg_mask": trg_mask,  # 这个还没有改变
            "right_next": right_next,  # 这个是没有变化之前的right_next
            "enc_cache": enc_cache,     # 这个不变
            'left_stop': dec_states['left_stop'],  # 这个没变
            'right_stop': dec_states['right_stop'] #这个没变
        }

        return outputs[:, -1, :], dec_states

    def forward(self, src, trg):
        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len]

        # ctx: [batch_size, src_
        # seq_len, context_size=enc_hid_dim*2]
        # ctx_mask: [batch_size, src_seq_len]

        ctx, ctx_mask = self.encoder(src)
        embeddings, hiddens, outputs, trg_mask, enc_cache, left_trg, right_trg, sorted_trg = self.init_decoder_when_train(
            ctx, ctx_mask, trg)

        # [batch_size, trg_seq_len, vocab_size]
        outputs, sorted_trg = self.decoder(left_trg, right_trg, ctx, ctx_mask, embeddings, hiddens, outputs, sorted_trg,
                                           enc_cache, trg, trg_mask)

        return outputs, sorted_trg

    def reorder_dec_states(self, dec_states, new_beam_indices, beam_size):
        # 相当于按照beam_size 重新排序, 现在的batch_size 其实是batch_size * beam_size

        hiddens = dec_states['hiddens']
        embeddings = dec_states['embeddings']
        outputs = dec_states['outputs']
        trg_mask = dec_states['trg_mask']
        right_next = dec_states['right_next']
        left_stop = dec_states['left_stop']
        right_stop = dec_states['right_stop']

        # 根据beam的结果重新处理

        batch_size = hiddens.size(0) // beam_size

        hiddens = tensor_gather_helper(gather_indices=new_beam_indices,
                                       gather_from=hiddens,
                                       batch_size=batch_size,
                                       beam_size=beam_size,
                                       gather_shape=hiddens.size())
        dec_states['hiddens'] = hiddens

        # 首先按照beam排序，后拼接上当前的word
        embeddings = tensor_gather_helper(gather_indices=new_beam_indices,
                                          gather_from=embeddings,
                                          batch_size=batch_size,
                                          beam_size=beam_size,
                                          gather_shape=embeddings.size())

        trg_mask = tensor_gather_helper(gather_indices=new_beam_indices,
                                        gather_from=trg_mask,
                                        batch_size=batch_size,
                                        beam_size=beam_size,
                                        gather_shape=trg_mask.size())

        final_word_indices = dec_states['final_word_indices']
        words = final_word_indices[:, :, -1].view(batch_size * beam_size)
        embed = self.decoder.embeddings(words).unsqueeze(1)
        _trg_mask = words.detach().eq(PAD) + words.detach().eq(EOS) + words.detach().eq(
            BOS) + words.detach().eq(MOS)

        if right_next:
            embeddings = torch.cat((embeddings, embed), dim=1)
            # beam_search中已经把这些词语调成了 PAD
            # if right_stop:
            #     _trg_mask = torch.ones(trg_mask.shape[0], dtype=torch.int8).cuda()
            trg_mask = torch.cat((trg_mask, _trg_mask.unsqueeze(1)), dim=1)
        else:
            embeddings = torch.cat((embed, embeddings), dim=1)
            # if left_stop:
            #     _trg_mask = torch.ones(trg_mask.shape[0], dtype=torch.int8).cuda()
            trg_mask = torch.cat((_trg_mask.unsqueeze(1), trg_mask), dim=1)
        trg_mask.requires_grad = False

        dec_states['embeddings'] = embeddings

        outputs = tensor_gather_helper(gather_indices=new_beam_indices,
                                       gather_from=outputs,
                                       batch_size=batch_size,
                                       beam_size=beam_size,
                                       gather_shape=outputs.size())
        dec_states['outputs'] = outputs

        trg_mask = tensor_gather_helper(gather_indices=new_beam_indices,
                                        gather_from=trg_mask,
                                        batch_size=batch_size,
                                        beam_size=beam_size,
                                        gather_shape=trg_mask.size())
        dec_states['trg_mask'] = trg_mask

        left_stop = tensor_gather_helper(gather_indices=new_beam_indices,
                                         gather_from=left_stop,
                                         batch_size=batch_size,
                                         beam_size=beam_size,
                                         gather_shape=left_stop.size())

        if not right_next:
            for i in range(left_stop.shape[0]):
                if words[i] == EOS:
                    left_stop[i] = 1

        dec_states['left_stop'] = left_stop

        right_stop = tensor_gather_helper(gather_indices=new_beam_indices,
                                          gather_from=right_stop,
                                          batch_size=batch_size,
                                          beam_size=beam_size,
                                          gather_shape=right_stop.size())

        if right_next:
            for i in range(right_stop.shape[0]):
                if words[i] == EOS:
                    right_stop[i] = 1
        dec_states['right_stop'] = right_stop

        dec_states['right_next'] = not right_next
        return dec_states
