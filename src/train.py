from imports_pytorch import * 
from transformer_model import *

loss_object = nn.CrossEntropyLoss(reduction="none")

def loss_function(real, pred):
    mask = (real != 0)

    loss_ = loss_object(real, torch.softmax(pred, dim=-1))

    mask  = mask.to(loss_.type())
    loss_ *= mask

    return torch.sum(loss_) / torch.sum(mask)

transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size, 
                          pe_input=input_vocab_size, 
                          pe_target=target_vocab_size,
                          rate=dropout_rate)

def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)

    dec_padding_mask = create_padding_mask(inp)

    look_ahead_mask  = create_look_ahead_mask(tar.shape[1])

    dec_target_padding_mask = create_padding_mask(tar)

    combined_mask = torch.max(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_out = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    prediction, _ = transformer(inp, tar_inp, enc_padding_mask, combined_mask, dec_padding_mask)
    loss          = loss_function(real, pred)
    loss.backward()
    optimizer.step()

def train(train_dataset, EPOCHS):
    for epoch in EPOCHS:
        start = time.time()
        
        for (batch, (inp, tar)) in enumerate(train_dataset):
            train_step(inp, tar)
            if batch%50==0:
                print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, batch, train_loss.result(), train_accuracy.result()))
              
        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                                ckpt_save_path))
            
        print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                        train_loss.result(), 
                                                        train_accuracy.result()))

        print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))