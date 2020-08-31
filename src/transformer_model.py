from imports_pytorch import *

class ConvEncoderLayer(nn.Module):
    def __init__(self):
        super(ConvEncoderLayer, self).__init__()
        res50_model = models.resnet50(pretrained=True)
        print(res50_model)
        print(*list(res50_model.children())[-3])
        self.res50_conv  = nn.Sequential(*list(res50_model.children())[:-4])
        # self.conv2       = nn.Conv2d(1024, 512, kernel_size=1)

    def forward(self, x):
        x = self.res50_conv(x)
        # x = self.conv2(x)
        print(x.size())
        x = x.view(x.size()[0], 512, -1)
        print(x.size())
        return x


## Transformers

def get_angles(pos, i, d_model):
    angles_rates = 1 / np.power(10000, (2*(i//2))/np.float32(d_model))
    return pos * angles_rates

def positional_encoding(position, d_model):
    angles_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    angles_rads[:, 0::2] = np.sin(angles_rads[:, 0::2])

    angles_rads[:, 1::2] = np.cos(angles_rads[:, 1::2])

    pos_encoding = angles_rads[np.newaxis, ...]

    return torch.tensor(pos_encoding).to(torch.float32)

def create_padding_mask(seq):
    seq = (seq == 0).to(torch.float32)
    return seq[:, None, None, :]

def create_look_ahead_mask(size):
    mask = torch.triu(torch.ones((size, size)), diagonal=1)
    return mask

def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = torch.matmul(q, k.transpose(2, 3))
    # print(matmul_qk)
    dk = torch.tensor(k.shape[-1]).to(torch.float32)
    scaled_attention_logistics = matmul_qk/torch.sqrt(dk)
    # print(scaled_attention_logistics)
    if mask is not None:
        scaled_attention_logistics += (mask * -1e9)
    attention_weight = torch.softmax(scaled_attention_logistics, axis=-1)
    # print(attention_weight)
    output = torch.matmul(attention_weight, v)

    return output, attention_weight

def print_out(q, k, v):
  temp_out, temp_attn = scaled_dot_product_attention(
      q, k, v, None)
  print ('Attention weights are:')
  print (temp_attn)
  print ('Output is:')
  print (temp_out)

class MultiHeadAttention(nn.Module):
    def __init__(self, inp_shape, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model   = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq    =  nn.Linear(inp_shape, d_model)
        self.wk    =  nn.Linear(inp_shape, d_model)
        self.wv    =  nn.Linear(inp_shape, d_model)

        self.dense =  nn.Linear(inp_shape, d_model)
    
    def split_heads(self, x, batch_size):
        x = torch.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return torch.transpose(x, 1, 2)

    def forward(self, v, k, q, mask):
        batch_size = q.shape[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        # print(q.shape, k.shape, v.shape)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = torch.transpose(scaled_attention, 1, 2)
        concat_attention = torch.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)

        return output, attention_weights
def point_wise_feed_forward_network(inp_shape, d_model, diff):
    return nn.Sequential(
        nn.Linear(inp_shape, diff),
        nn.ReLU(),
        nn.Linear(diff, d_model)
    )

class EncoderLayer(nn.Module):
    def __init__(self, inp_shape, d_model, num_heads, diff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(inp_shape, d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(inp_shape, d_model, diff)

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1    = nn.Dropout(rate)
        self.dropout2    = nn.Dropout(rate)

    def forward(self, x, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output    = self.dropout1(attn_output)
        out1           = self.layernorm1(x + attn_output)

        fnn_output     = self.ffn(out1)
        fnn_output     = self.dropout2(fnn_output)
        out2           = self.layernorm2(out1 + fnn_output)

        return out2

class DecoderLayer(nn.Module):
    def __init__(self, inp_shape, d_model, num_heads, diff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mh1 = MultiHeadAttention(inp_shape, d_model, num_heads)
        self.mh2 = MultiHeadAttention(inp_shape, d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(inp_shape, d_model, diff)

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
        self.dropout3 = nn.Dropout(rate)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mh1(x, x, x, look_ahead_mask)
        attn1                      = self.dropout1(attn1)
        out1                       = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mh2(enc_output, enc_output, out1, padding_mask)
        attn2                      = self.dropout2(attn2)
        out2                       = self.layernorm2(attn2 + out1)

        fnn_output                 = self.ffn(out2)
        fnn_output                 = self.dropout3(fnn_output)
        out3                       = self.ffn(fnn_output + out2)
        return out3, attn_weights_block1, attn_weights_block2


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = torch.tensor(d_model).to(torch.float32)
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)
        self.enc_layers = [EncoderLayer(d_model, d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout    = nn.Dropout(rate)

    def forward(self, x, mask):
        seq_len = x.shape[1]

        x = self.embedding(x)
        print(x.shape)
        x *= torch.sqrt(self.d_model)
        x += self.pos_encoding[:, :seq_len, :]
        
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask)

        return x

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()
        self.d_model    = torch.tensor(d_model).to(torch.float32)
        self.num_layers = num_layers
        self.embedding  = nn.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        self.dec_layers   = [DecoderLayer(d_model, d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout      = nn.Dropout(rate)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        seq_len = x.shape[1]
        attention_weight = {}

        x = self.embedding(x)
        x*= torch.sqrt(self.d_model)
        x+= self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, look_ahead_mask, padding_mask)
            attention_weight['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weight['decoder_layer{}_block2'.format(i+1)] = block2
        return x, attention_weight

class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)
        self.final_layer = nn.Linear(d_model, target_vocab_size)
    def forward(self, inp, tar, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, enc_padding_mask)
        dec_output, attention_weights = self.decoder(tar, enc_output, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)
        return final_output, attention_weights
if __name__ == "__main__":
    # net = ConvEncoderLayer()
    # net(torch.zeros((1, 3, 32, 112)))
    # pos_encoding = positional_encoding(50, 512)
    # print(pos_encoding.shape)
    # x = torch.tensor([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
    # print(create_padding_mask(x))
    # x = torch.rand((1,3))
    # print(create_look_ahead_mask(x.shape[1]))
    # temp_k = torch.tensor([[10,0,0],
    #                   [0,10,0],
    #                   [0,0,10],
    #                   [0,0,10]]).to(dtype=torch.float32)  # (4, 3)

    # temp_v = torch.tensor([[   1,0],
    #                     [  10,0],
    #                     [ 100,5],
    #                     [1000,6]]).to(dtype=torch.float32)  # (4, 2)

    # This `query` aligns with the second `key`,
    # so the second `value` is returned.
    # temp_q = torch.tensor([[0, 10, 0]]).to(dtype=torch.float32)  # (1, 3)
    # print_out(temp_q, temp_k, temp_v)
    # temp_mha = MultiHeadAttention(inp_shape=512, d_model=512, num_heads=8)
    # y = torch.rand((1, 60, 512))
    # out, attn = temp_mha(y, k=y, q=y, mask=None)
    # print(out.shape, attn.shape)

    # sample_ffn = point_wise_feed_forward_network(512, 512, 2048)
    # sample_ffn(torch.rand((64, 50, 512))).shape
    # sample_encoder_layer = EncoderLayer(512, 512, 8, 2048)

    # sample_encoder_layer_output = sample_encoder_layer(
    #     torch.zeros((64, 43, 512)), None)

    # sample_decoder_layer = DecoderLayer(512, 512, 8, 2048)

    # sample_encoder_layer_output = sample_decoder_layer(
    #     torch.zeros((64, 50, 512)), sample_encoder_layer_output,None, None)
    # print(sample_encoder_layer_output.shape)  # (batch_size, input_seq_len, d_model)
    # sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8, 
    #                      dff=2048, input_vocab_size=8500,
    #                      maximum_position_encoding=10000)

    # temp_input =  ((torch.rand(64,62)*100)%100).to(dtype=torch.long)
    # # tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)

    # sample_encoder_output = sample_encoder(temp_input, mask=None)

    # sample_decoder = Decoder(num_layers=2, d_model=512, num_heads=8, 
    #                         dff=2048, target_vocab_size=8000,
    #                         maximum_position_encoding=5000)


    # temp_input =  ((torch.rand(64,26)*100)%100).to(dtype=torch.long)
    # output, attn = sample_decoder(temp_input, 
    #                             enc_output=sample_encoder_output, 
    #                             look_ahead_mask=None, 
    #                             padding_mask=None)

    # output.shape, attn['decoder_layer2_block2'].shape
    # print (sample_encoder_output.shape)  # (batch_size, input_seq_len, d_model)


    sample_transformer = Transformer(
    num_layers=2, d_model=512, num_heads=8, dff=2048, 
    input_vocab_size=8500, target_vocab_size=8000, 
    pe_input=10000, pe_target=6000)

    temp_input =  ((torch.rand(64,38)*100)%100).to(dtype=torch.long)
    temp_target =  ((torch.rand(64,36)*100)%100).to(dtype=torch.long)
    # temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
    # temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)

    fn_out, _ = sample_transformer(temp_input, temp_target, 
                                enc_padding_mask=None, 
                                look_ahead_mask=None,
                                dec_padding_mask=None)

    print(fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)