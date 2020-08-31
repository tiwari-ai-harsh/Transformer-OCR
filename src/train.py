from imports_pytorch import * 
from transformer_model import *
from dataset import WordDataset
from lr_schedular import CoustomLR

loss_object = nn.CrossEntropyLoss(reduction="none")

def loss_function(real, pred):
    mask = (real != 0)
    loss_ = loss_object(real, torch.softmax(pred, dim=-1))
    mask  = mask.to(loss_.type())
    loss_ *= mask

    return torch.sum(loss_) / torch.sum(mask)

wd = WordDataset(transform=transforms.Compose([transforms.Resize(size=(IMAGE_HEIGHT, IMAGE_WIDTH)), transforms.ToTensor()]))
dataloader = DataLoader(wd, batch_size=8, shuffle=True, num_workers=4)

# transformer = Transformer(num_layers, d_model, num_heads, dff,
#                           input_vocab_size, target_vocab_size, 
#                           pe_target=target_vocab_size,
#                           rate=dropout_rate)

sample_transformer = Transformer(num_layers, d_model, num_heads, dff, 
                            target_vocab_size=wd.vocab_size, 
                            pe_input=1000, pe_target=1000).to(device)

learning_rate = 0.0001
optimizer = optim.Adam(sample_transformer.parameters(), lr=learning_rate, betas=[0.9, 0.98], eps=1e-9)
schedular = CoustomLR(optimizer, d_model)
# data, target = wd[1]
# print(data.unsqueeze(dim=0).shape)
# print(target.unsqueeze(dim=0).shape)

# fn_out, _ = sample_transformer(data.unsqueeze(dim=0), target.unsqueeze(dim=0), 
#                             enc_padding_mask=None, 
#                             look_ahead_mask=None,
#                             dec_padding_mask=None)

# print(fn_out.shape)
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


sample_transformer = sample_transformer.to(device)
for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, tar = data
        # print(inputs.shape)
        # print(tar.shape)
        # break
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        # print(tar_inp.shape)
        # print(tar_real.shape)
        # break
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inputs, tar_inp)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        predictions, _ = sample_transformer(inputs.to(device), tar_inp.to(device), 
                                 enc_padding_mask.to(device), 
                                 combined_mask.to(device), 
                                 dec_padding_mask.to(device))

        loss = loss_function(tar_real.to(device), predictions)
        loss.backward()
        optimizer.step()
        schedular.step()

        # print statistics
        running_loss += loss.item()
        # if i % 100 == 99:    # print every 2000 mini-batches
        if i % 100 == 99:    # print every 2000 mini-batches
            print("outputs: ", outputs)
            print("labels: ", labels)
            torch.save(model.state_dict(), (PATH+"{}_.pt".format(i)))
            print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 100))
    running_loss = 0.0

print('Finished Training')


# def train(train_dataset, EPOCHS):
#     for epoch in EPOCHS:
#         start = time.time()
        
#         for (batch, (inp, tar)) in enumerate(train_dataset):
#             train_step(inp, tar)
#             if batch%50==0:
#                 print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
#                     epoch + 1, batch, train_loss.result(), train_accuracy.result()))
              
#         if (epoch + 1) % 5 == 0:
#             ckpt_save_path = ckpt_manager.save()
#             print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
#                                                                 ckpt_save_path))
            
#         print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
#                                                         train_loss.result(), 
#                                                         train_accuracy.result()))

#         print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))