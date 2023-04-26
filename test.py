import logging
import os
import matplotlib.pyplot as plt
import numpy as np

a = np.arange(20).reshape((4, 5))
plt.imshow(a)
plt.show()

# import time
# a=time.asctime()
# print(a)
# print(type(a))

# def test(d=1):
#     figure=plt.figure(figsize=(10, 10))
#     x = [i for i in range(100)]
#     y = [i*d for i in range(100)]
#     plt.plot(x, y)
#     return figure
#
# figure = test(d=1)
# plt.savefig('111.png')
# figure = test(d=100)
# plt.savefig('100.png')

# from tqdm import tqdm
# a=(i for i in range(100000))
# for i in tqdm(a):
#     print(i)

# outfile = open('./output/seg/debug/debug1\\log.txt', 'w', encoding='utf-8')
# a=1
# outfile.write(f'111111111 {a}\n')
# outfile.write(f'111111111 {a}\n')
# outfile.write(f'111111111 {a}\n')
# outfile.close()

# import numpy as np
# np.random.seed(0)
# print(np.random.randint(10))
# print(np.random.randint(10))
# print(np.random.randint(10))
# print(np.random.randint(10))
# print(np.random.randint(10))
# print(np.random.randint(10))
# print(np.random.randint(10))


# from util import loss
# a = getattr(loss,'BendingEnergyLoss')
# print(dir(loss))

# seg_net.train()
# seg_train_loss = []
# seg_train_metric = []
#
# for i in tqdm(range(config['TrainConfig']['Seg']['seg_step_per_epoch'])):
#     id, volume, label = next(iter(seg_train_dataloader))
#     volume = volume.to(seg_device)
#     label = label.to(seg_device)
#
#     seg_optimizer.zero_grad()
#
#     predict = seg_net(volume)
#
#     predict_softmax = F.softmax(predict, dim=1)
#
#     axis_order = (0, label.dim() - 1) + tuple(range(1, label.dim() - 1))
#     label_one_hot = F.one_hot(label.squeeze(dim=1).long()).permute(axis_order).contiguous()
#     loss = seg_loss_function(predict_softmax, label_one_hot.float())
#
#     loss.backward()
#
#     seg_optimizer.step()
#
#     seg_train_loss.append(loss.item())
#
#     predict_one_hot = F.one_hot(torch.argmax(predict_softmax, dim=1).long()).permute(axis_order).contiguous()
#
#     dice = dice_metric(predict_one_hot.clone().detach(), label_one_hot.clone().detach(), background=True)
#
#     seg_train_metric.append(dice.item())
# seg_mean_train_loss = np.mean(seg_train_loss)
# seg_mean_train_metric = np.mean(seg_train_metric)
# print(f'seg training loss: {seg_mean_train_loss}')
# print(f'seg training dice:  {seg_mean_train_metric}')
# writer.add_scalar("train/seg/loss", seg_mean_train_loss, epoch)
# writer.add_scalar("train/seg/dice", seg_mean_train_metric, epoch)
# writer.add_scalar("lr/seg", seg_optimizer.get_cur_lr(), epoch)
# if epoch % config['TrainConfig']['Seg']['val_interval'] == 0:
#     seg_net.eval()
#     seg_val_losses = []
#     seg_val_metrics = []
#     for id, volume, label in tqdm(seg_val_dataloader):
#         volume = volume.to(seg_device)
#         label = label.to(seg_device)
#
#         predict = seg_net(volume)
#
#         predict_softmax = F.softmax(predict, dim=1)
#
#         axis_order = (0, label.dim() - 1) + tuple(range(1, label.dim() - 1))
#         label_one_hot = F.one_hot(label.squeeze(dim=1).long()).permute(axis_order).contiguous()
#
#         loss = seg_loss_function(predict_softmax, label_one_hot.float())
#         seg_val_losses.append(loss.item())
#
#         predict_one_hot = F.one_hot(torch.argmax(predict, dim=1).long()).permute(axis_order).contiguous()
#
#         dice = dice_metric(predict_one_hot.clone().detach(), label_one_hot.clone().detach(), background=True)
#
#         seg_val_metrics.append(dice.item())
#
#         predict_argmax = torch.argmax(predict_softmax, dim=1, keepdim=True)
#         tensorboard_visual_segmentation(mode='val/seg', name=id[0], writer=writer, step=epoch,
#                                         volume=volume[0].clone().detach(),
#                                         predict=predict_argmax[0].clone().detach(),
#                                         target=label[0].clone().detach())
#     seg_val_mean_loss = np.mean(seg_val_losses)
#     seg_val_mean_metric = np.mean(seg_val_metrics)
#     print(f'seg val loss: {seg_val_mean_loss}')
#     print(f'seg val dice:  {seg_val_mean_metric}')
#     writer.add_scalar("val/seg/loss", seg_val_mean_loss, epoch)
#     writer.add_scalar("val/seg/dice", seg_val_mean_metric, epoch)
#
#     if seg_val_mean_metric > seg_best_val_metric:
#         seg_best_val_metric = seg_val_mean_metric
#         model_saver.save(os.path.join(basedir, "checkpoint", 'best_epoch_' + str(epoch).zfill(4) + ".pth"),
#                          {"model": seg_net.state_dict(), "optim": seg_optimizer.state_dict()})
