import os
import torch
from torch.autograd import Variable
from Auto_enco import MyAuto
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image
from C3D_model import C3D
import numpy as np
from regression import Regression
import matplotlib.pyplot as plt

torch.cuda.set_device(1)
torch.manual_seed(1)

# Super parameters
EPOCH = 100
BATCH_SIZE = 1
LR = 1e-3
is_shuffle = False


def get_raw_transformers():
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def get_scale_transformers(size):
    return transforms.Compose([transforms.Scale((size, size)), transforms.ToTensor(),
                               transforms.Normalize(mean=[0.484, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".bmp"])


def load_img(file_path):
    try:
        img = Image.open(file_path).convert('RGB')
        return img
    except Exception as e:

        print e
        return None


class ImageFolderDataSets(torch.utils.data.Dataset):
    def __init__(self, image_dir, input_transform, loader=load_img):
        self.image_dir = image_dir
        source_image_names = [x for x in os.listdir(image_dir)]
        no_list = [int(f[:-4]) for f in source_image_names]
        no_list_sorted = sorted(no_list)
        recon_image_name = [str(No) + ".jpg" for No in no_list_sorted]
        self.image_name = recon_image_name
        self.input_transform = input_transform
        self.loader = loader

    def __getitem__(self, index):
        img_name = self.image_name[index]
        path = os.path.join(self.image_dir, img_name)
        image = self.loader(path)

        if image is not None:
            image = self.input_transform(image)
            return image

    def __len__(self):
        return len(self.image_name)

    def name(self):
        return self.image_name


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 128, 128)
    return x


def main():

    if not os.path.exists('./concat_img'):
        os.mkdir('./concat_img')
    img_dir = "/home/disk/duli/My_autoencoder/data/driving_data/train"
    tran = get_scale_transformers(128)
    data_set = ImageFolderDataSets(img_dir, tran)
    print len(data_set)
    print data_set.name()
    total_images = data_set.__len__()
    #print total_images
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    #print len(data_loader)
    #print data_set[0].shape
    # use trained auto_encoder model
    auto_encoder = MyAuto()
    auto_encoder.load_state_dict(torch.load('nextsteering_autoencoder.pth'))
    auto_encoder.cuda()
    #optimizer = torch.optim.Adam(auto_encoder.parameters(), lr=LR)
    loss_function = torch.nn.MSELoss()
    # initialize the C3D
    net = C3D()
    net.cuda()
    net.eval()
    # Steering ground truth with initialize regression
    with open('./angle.txt', 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    #label_length = labels.__len__()
    # Regression initialization
    model = Regression(192, 10, 1)
    model.cuda()
    ste_optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    loss_func = torch.nn.MSELoss()
    plt.ion()
    generated_steering = []  # save predicted steering angle values
    train_loss = 0
    # Training
    for epoch in range(EPOCH):
        for step, img in enumerate(data_loader):
            b_x = Variable(img)
            b_x = b_x.view(-1, 16384)
            b_x = b_x.cuda()
            ste_truth = torch.FloatTensor([float(labels[step])])
            ste_truth = Variable(ste_truth).cuda()
            # print ste_truth
            if step == 0:
                # Appearance prediction(1)
                b_wait = torch.zeros(img.shape)
                b_wait = Variable(b_wait)
                b_wait = b_wait.view(-1, 16384)
                b_wait = b_wait.cuda()
                encoded, decoded = auto_encoder(b_wait)
                #print encoded.size()
                #loss = loss_function(decoded, b_x)
                b_wait = b_x
                #optimizer.zero_grad()
                #loss.backward()
                #optimizer.step()
                pic = to_img(decoded.cpu().data)
                if epoch == EPOCH:
                    save_image(pic, './concat_img/image_{}.jpg'.format(step))
                # Steering angle prediction
                e_wait = torch.rand(6, 8, 8)
                r_wait = torch.rand(3, 8, 8)
                #print a.size()
                # reconstruction representation layer
                wait = torch.cat([e_wait, r_wait], 0)
                #print wait.size()
                wait_view = wait.view(1, 9, 1, 8, 8)
                wait_view = Variable(wait_view)
                wait_view = wait_view.cuda()
                recon = net(wait_view)     # C3D  9*8*8 -> 3*8*8(R_t)
                r_wait = recon
                #print("recon.size = {}".format(recon.size()))
                # Steering angle regression Regression 3*8*8->1
                ste = recon.view(1, 192)
                # print model
                prediction = model(ste)
                #print prediction
                #print e.size()
                ste_loss = loss_func(prediction, ste_truth)
                ste_optimizer.zero_grad()
                #print ('loss:{:.4f}'.format(ste_loss.data[0]))
                ste_loss.backward()
                ste_optimizer.step()
                generated_steering.append(prediction)

            if step > 0 :
                # apperance prediction(2)
                encoded, decoded = auto_encoder(b_wait)
                pic = to_img(decoded.cpu().data)
                if epoch == EPOCH:
                    save_image(pic, './concat_img/image_{}.jpg'.format(step))
                # Steering angle prediction(2)
                a, b = auto_encoder(b_x)  # 3*8*8
                a_wait, b_wa = auto_encoder(b_wait)
                e_p = a_wait - a
                e_n = a - a_wait
                e = torch.cat([e_p, e_n], 0)
                e_wait = e
                # reconstruction representation layer
                wait = torch.cat([e_wait, r_wait], 0)
                #print wait.size()  # (9, 8, 8)
                wait_view = wait.view(1, 9, 1, 8, 8)
                wait_view = wait_view.cuda()
                recon = net(wait_view)  # C3D  9*8*8 -> 3*8*8
                #print("recon2.size = {}".format(recon.size()))
                # Steering angle regression Regression 3*8*8->1
                ste = recon.view(1, 192)
                # print model
                prediction = model(ste)
                #print prediction
                # print e.size()
                ste_loss = loss_func(prediction, ste_truth)
                train_loss += ste_loss.data[0]
                ste_optimizer.zero_grad()
                #print ('loss:{:.4f}'.format(ste_loss.data[0]))
                #ste_loss.backward()
                ste_optimizer.step()
                generated_steering.append(prediction)
                r_wait = recon
                b_wait = b_x
                '''
                if step % 109 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch,
                        step * len(b_x),
                        len(data_loader.dataset), 1. * step / len(data_loader),
                        ste_loss.data[0] / len(img)))
                '''
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(data_loader.dataset)))
        torch.save(auto_encoder.state_dict(), './concat_autoencoder.pth')
        torch.save(auto_encoder, 'concat_model.pkl')
    fileobject = open('predicted_steering.txt', 'w')
    for ip in generated_steering:
        fileobject.write(ip)
        fileobject.write('\n')
    fileobject.close()
if __name__ == '__main__':
    main()