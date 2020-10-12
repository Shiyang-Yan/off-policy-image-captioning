from seq_auto_image import *
import torch
import random
import numpy as np
class Seq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(8668, 256, 256, 512, 0.1)
        self.attention = Attention(512, 512)
        self.decoder = Decoder(8668, 512, 512, 512, 0.1, self.attention)
        self.init_hidden = nn.Linear(2048, 512)

    def forward(self, trg, image_features, teacher_forcing_ratio=0.5):
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
        src = trg
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        _, hidden_encoder  = self.encoder(src)
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).cuda()
        hidden = hidden_encoder
        #hidden = F.relu(self.init_hidden(image_features.mean(0)))

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]
        for t in range(1, trg_len):
            # insert input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, hidden_encoder, image_features)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1
        return outputs

    def sample_greedy(self,  trg, image_features, ix_to_word):
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
        src = trg
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outcap = np.empty((batch_size, 0)).tolist()
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).cuda()
        _, hidden_encoder  = self.encoder(src)
        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        hidden = hidden_encoder
        #hidden = F.relu(self.init_hidden(image_features.mean(0)))
      # first input to the decoder is the <sos> tokens
        input = trg[0, :]
        for t in range(1, 181):
            # insert input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden,hidden_encoder, image_features)

            # place predictions in a tensor holding predictions for each token
            outputs[t-1] = output

            # decide if we are going to use teacher forcing or not

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = top1
            for k in range(batch_size):
                index = top1[k]
                #if index > 0 and index != 8667 and index != 8666:
                #    word = ix_to_word[str(int(index))]
                outcap[k].append(index)
        return outcap




    def sample(self, trg, image_features, ix_to_word):
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        src = trg
        trg_vocab_size = self.decoder.output_dim
        outcap = np.empty((batch_size, 0)).tolist()
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).cuda()
        logprobs_all = torch.zeros(batch_size, trg_len)
        sampledids = torch.zeros(batch_size, trg_len)
        output_all = torch.zeros(trg_len, batch_size, trg_vocab_size)
        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        # first input to the decoder is the <sos> tokens
        _, hidden_encoder  = self.encoder(src)
        hidden = hidden_encoder
        #hidden = F.relu(self.init_hidden(image_features.mean(0))).cuda()
        hiddens_all = torch.zeros((batch_size, trg_len, 512)).cuda()    
        input = trg[0, :]
        for t in range(1, 180):
            # insert input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, hidden_encoder, image_features)
            # place predictions in a tensor holding predictions for each token
            hiddens_all[:,t-1,:] = hidden.squeeze()
            outputs[t-1] = output

            # decide if we are going to use teacher forcing or not

            # get the highest predicted token from our predictions
            output = F.log_softmax(output, 1)
            output_all[t] =  output
            prob = torch.exp(output)
            it = torch.multinomial(prob, 1)
            sampledLogprobs = output.gather(1, it)
            logprobs_all[:, t]= sampledLogprobs.squeeze()
            sampledids[:, t-1] = it.squeeze()
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = it.squeeze()

            for k in range(batch_size):
                index = it[k]
                if index > 0 and index != 8667:
                    word = ix_to_word[str(int(index))]
                    outcap[k].append(word)
        return outcap, sampledids, logprobs_all, hiddens_all, outputs, output_all


    def sample_greedy_trigram(self, trg,image_features, ix_to_word):
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
        sampled_ids = []
        src = trg
        _, hidden_encoder  = self.encoder(src)
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outcap = np.empty((batch_size, 0)).tolist()
        # tensor to store decoder outputs
        seq = torch.zeros((batch_size, trg_len), dtype=torch.long)
        seqLogprobs = torch.zeros(batch_size, trg_len)
        trigrams = []
        hidden = hidden_encoder
        #hidden = F.relu(self.init_hidden(image_features.mean(0)))
        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]
        for t in range(1, 181):
            # insert input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            if t == 1:
                output, hidden = self.decoder(input, hidden, hidden_encoder, image_features)
                logprobs = F.log_softmax(output, 1)
                it = logprobs.argmax(1)
            if t > 1:
                #if t == 2:
                #    unfinished = it > 0
                #else:
                #    unfinished = unfinished * (it > 0)
                #if unfinished.sum() == 0:
                #    break
                #it = it * unfinished.type_as(it)
                seq[:,t-1] = it
                input = it
                sampled_ids.append(it)
                seq[:, t-1] = it
                output, hidden = self.decoder(input, hidden, hidden_encoder, image_features)
                logprobs = F.log_softmax(output, 1)
            if t >= 4:
                # Store trigram generated at last step
                prev_two_batch = seq[:, t-3:t-1]
                for i in range(batch_size):  # = seq.size(0)
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    current = seq[i][t-1]
                    if t == 4:  # initialize
                        trigrams.append({prev_two: [current]})  # {LongTensor: list containing 1 int}
                    elif t > 4:
                        if prev_two in trigrams[i]:  # add to list
                            trigrams[i][prev_two].append(current)
                        else:  # create list
                            trigrams[i][prev_two] = [current]
                # Block used trigrams at next step
                prev_two_batch = seq[:, t-2:t]
                mask = torch.zeros(logprobs.size(), requires_grad=False).cuda()  # batch_size x vocab_size
                for i in range(batch_size):
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    if prev_two in trigrams[i] and prev_two != (0,0):
                        for j in trigrams[i][prev_two]:
                            mask[i, j] += 1
                # Apply mask to log probs
                # logprobs = logprobs - (mask * 1e9)
                alpha = 2  # = 4
                logprobs = logprobs + (mask * -0.693 * alpha)  # ln(1/2) * alpha (alpha -> infty works best# )
            it = logprobs.argmax(1)
            for k in range(batch_size):
                index = it[k]
                if index > 0 and index != 8667:
                    word = ix_to_word[str(int(index))]
                    outcap[k].append(word)
        return outcap
'''

    def sample_greedy_trigram(self, captions, ix_to_word, states = None):
        batch_size = captions.size(0)
        outcap = np.empty((captions.size(0), 0)).tolist()
        sampled_ids = []
        features = self.encoder(captions.long())
        inputs = features.unsqueeze(1)
        seq = torch.zeros((captions.size(0), self.decoder.max_seg_length), dtype=torch.long)
        seqLogprobs = torch.zeros(captions.size(0), self.decoder.max_seg_length)
        trigrams = []
        for t in range(self.decoder.max_seg_length):
            if t == 0:
                hiddens, states = self.decoder.lstm(inputs, states)
                outputs = self.decoder.linear(hiddens.squeeze(1))  # o
                first_logprobs = F.log_softmax(outputs, 1)
                sampleLogprobs, it = first_logprobs.max(1)  # predicted: (b
            if t > 0:
                sampleLogprobs, it = logprobs.max(1)  # predicted: (batch_size)
                sampled_ids.append(it)
            if t >= 1:
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                if unfinished.sum() == 0:
                    break
                it = it * unfinished.type_as(it)
                seq[:, t-1] = it
                seqLogprobs[:, t-1] = sampleLogprobs.view(-1)

            hiddens, states = self.decoder.lstm(inputs, states)  # hiddens: (batch_size, 1, hidden_size)
            outputs = self.decoder.linear(hiddens.squeeze(1))  # outputs:  (batch_size, vocab_size)
            logprobs = F.log_softmax(outputs, 1)
            if  t >= 3:
                # Store trigram generated at last step
                prev_two_batch = seq[:, t-3:t-1]
                for i in range(batch_size):  # = seq.size(0)
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    current = seq[i][t - 1]
                    if t == 3:  # initialize
                        trigrams.append({prev_two: [current]})  # {LongTensor: list containing 1 int}
                    elif t > 3:
                        if prev_two in trigrams[i]:  # add to list
                            trigrams[i][prev_two].append(current)
                        else:  # create list
                            trigrams[i][prev_two] = [current]
                # Block used trigrams at next step
                prev_two_batch = seq[:, t-2:t]
                mask = torch.zeros(logprobs.size(), requires_grad=False).cuda()  # batch_size x vocab_size
                for i in range(batch_size):
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    if prev_two in trigrams[i]:
                        for j in trigrams[i][prev_two]:
                            mask[i, j] += 1
                # Apply mask to log probs
                # logprobs = logprobs - (mask * 1e9)
                alpha = 2.0  # = 4
                logprobs = logprobs + (mask * -0.693 * alpha)  # ln(1/2) * alpha (alpha -> infty works best)
            inputs = self.decoder.embed(it.cuda())  # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)  # inputs: (batch_size, 1, embed_size)
            for k in range(batch_size):
                index = seq[k][t]
                if index > 0 and index != 8667 and index != 8666:
                    word = ix_to_word[str(int(index))]
                    outcap[k].append(word)
        return outcap


    def sample(self, captions, ix_to_word, states=None):
            logprob_all = torch.zeros((captions.size(0), self.decoder.max_seg_length))
            outcap = np.empty((captions.size(0), 0)).tolist()
            sampled_ids = []
            features = self.encoder(captions.long())
            inputs = features.unsqueeze(1)
            hiddens_all = torch.zeros(captions.size(0), 512, self.decoder.max_seg_length)
            for i in range(self.decoder.max_seg_length):
                hiddens, states = self.decoder.lstm(inputs, states)  # hiddens: (batch_size, 1, hidden_size)
                hiddens_all[:,:,i] = hiddens.squeeze()
                outputs = self.decoder.linear(hiddens.squeeze(1))  # outputs:  (batch_size, vocab_size)
                logprob = F.log_softmax(outputs, 1)
                prob = torch.exp(logprob)
                it = torch.multinomial(prob, 1)
                sampledLogprobs = logprob.gather(1, it)
                logprob_all[:,i] = sampledLogprobs.squeeze()
                sampled_ids.append(it.squeeze())
                inputs = self.decoder.embed(it.squeeze())  # inputs: (batch_size, embed_size)
                inputs = inputs.unsqueeze(1)  # inputs: (batch_size, 1, embed_size)
                for k in range(inputs.size(0)):
                    index = it[k]
                    if index > 0 and index != 8667 and index != 8666:
                        word = ix_to_word[str(int(index))]
                        outcap[k].append(word)
            sampled_ids = torch.stack(sampled_ids, 1)  # sampled_ids: (batch_size, max_seq_length)
            return outcap, logprob_all, hiddens_all, sampled_ids

'''
