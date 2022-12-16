# import time
#
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
#
# from data import get_dataset, get_dataset_info
# from sharpening import sharpening_loss_scaler, sharpening_loss
# from constants import MODEL_PATH
# from models import MiniFCNet, BasicCNN1Net, BasicCNN2Net, VGG
#
# DATASET = "CIFAR10"
# LOAD_MODEL = False
#
# BATCH_SIZE = 32
# EPOCHS = 10
#
#
# def main():
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#
#     print(f"device: {device}")
#
#     train_set, val_set, test_set = get_dataset(dataset=DATASET, validation_set_pc=0.1)
#     image_dim, n_outputs, classes = get_dataset_info(dataset=DATASET)
#     print()
#     train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)
#     val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)
#     test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=3)
#
#     # Model.
#     net = MiniFCNet(image_dim=image_dim, n_outputs=n_outputs)
#     # net = BasicCNN2Net(n_outputs)
#     # net = VGG("VGG11")
#     net.to(device)
#     model_path = MODEL_PATH / net.filepath
#
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
#     # optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=0.0001)
#
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
#     # add warmup, cool down, cyclic cosine scheduler, noise?, random restart?, ..?
#
#     if LOAD_MODEL:
#         net = MiniFCNet(image_dim=image_dim, n_outputs=n_outputs)
#         net.load_state_dict(torch.load(model_path))
#         net.to(device)
#     else:
#         for epoch in range(EPOCHS):
#             train(epoch, net, device, train_loader, optimizer, criterion)
#             validate(net, device, val_loader)
#             scheduler.step()
#
#         print('Finished Training')
#         torch.save(net.state_dict(), model_path)
#         print("Model saved")
#
#     test(net, device, test_loader, classes)
#
#
# def train(epoch, net, device, train_loader: DataLoader,
#           optimizer: torch.optim.Optimizer, criterion):
#     net.train()
#     running_loss = 0.0
#     for batch_idx, data in enumerate(train_loader, 0):
#         inputs, labels = data[0].to(device), data[1].to(device)
#         optimizer.zero_grad()
#
#         outputs, activations_ls = net(inputs)
#         # Loss
#         objective_loss = criterion(outputs, labels)
#         batches_in_loader = len(train_loader)
#         sharpening_scaler = sharpening_loss_scaler(epoch, batch_idx,
#                                                    batches_in_loader,
#                                                    BATCH_SIZE, EPOCHS)
#         loss = objective_loss + sharpening_scaler * sharpening_loss(activations_ls)
#
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#         log_period = 300
#         if batch_idx % log_period == (log_period-1):  # print every 2000 mini-batches
#             print(f'[{epoch + 1}, {batch_idx + 1:5d}] loss: {running_loss / (log_period - 1):.3f}')
#             running_loss = 0.0
#
#
# def validate(net, device, val_loader):
#     net.eval()
#     with torch.no_grad():
#         acc_ls = []
#         correct = 0
#         total = 0
#         for i, data in enumerate(val_loader):
#             images, labels = data[0].to(device), data[1].to(device)
#             outputs, _ = net(images)
#             _, y = torch.max(outputs, dim=1)  # y - predictions.
#
#             correct += (y == labels).sum().item()
#             total += len(y)
#
#     print(f"val: {correct} / {total} = {correct/total : 0.3f}")
#
#
# def test(net, device, test_loader: DataLoader, classes):
#     net.eval()
#     # Test
#     correct_pred = {classname: 0 for classname in classes}
#     total_pred = {classname: 0 for classname in classes}
#
#     with torch.no_grad():
#         for data in test_loader:
#             images, labels = data[0].to(device), data[1].to(device)
#             outputs, _ = net(images)
#             _, predictions = torch.max(outputs, 1)
#             # collect the correct predictions for each class
#             for label, prediction in zip(labels, predictions):
#                 if label == prediction:
#                     correct_pred[classes[label]] += 1
#                 total_pred[classes[label]] += 1
#
#     # print accuracy for each class
#     for classname, correct_count in correct_pred.items():
#         accuracy = 100 * float(correct_count) / total_pred[classname]
#         print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
#
#     total = sum([count for count in total_pred.values()])
#     correct = sum([count for count in correct_pred.values()])
#     print(f'Accuracy of the network on the {total} test images: {100 * correct // total} %')
#
#
# if __name__ == '__main__':
#     torch.set_printoptions(linewidth=200)
#     main()
