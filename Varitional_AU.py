import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import os
from PIL import Image

if not os.path.exists('./vae_img'):
    os.mkdir('./vae_img')


def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 128, 128)
    return x


num_epochs = 100
batch_size = 128
learning_rate = 0.000001

# DATA SET


def get_raw_transformers():
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def get_scale_transformers(size):
    return transforms.Compose([transforms.Scale((size, size)), transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


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


img_dir = "/home/disk/duli/My_autoencoder/data/driving_data/train"
tran = get_scale_transformers(128)
data_set = ImageFolderDataSets(img_dir, tran)
print len(data_set)
total_images = data_set.__len__()
print total_images
data_loader = torch.utils.data.DataLoader(data_set, batch_size=27, shuffle=False, num_workers=4)
print len(data_loader)
print data_set[0].shape


# Model Class
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(16384, 4096)
        self.fc21 = nn.Linear(4096, 1024)
        self.fc22 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 4096)
        self.fc4 = nn.Linear(4096, 16384)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE()
if torch.cuda.is_available():
    model.cuda()

reconstruction_function = nn.MSELoss(size_average=False)


def loss_function(recon_x, x, mu, logvar):
    BCE = reconstruction_function(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.05)
    # KL divergence
    return BCE + KLD


optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(data_loader):
        img = data
        img = img.view(-1, 16384)
        img = Variable(img)
        if torch.cuda.is_available():
            img = img.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(img)
        loss = loss_function(recon_batch, img, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % 27 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(img),
                len(data_loader.dataset), 100. * batch_idx / len(data_loader),
                loss.data[0] / len(img)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(data_loader.dataset)))
    if epoch % 10 == 0:
        save = to_img(recon_batch.cpu().data)
        save_image(save, './vae_img/image_{}.jpg'.format(epoch))
torch.save(model.state_dict(), './vae.pth')