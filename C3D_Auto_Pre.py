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

if not os.path.exists('./c3d_vae_img'):
    os.mkdir('./c3d_vae_img')


def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 128, 128)
    return x


num_epochs = 100
batch_size = 27
learning_rate = 1e-3

# ,,,,,,,,,,,,DATA SET,,,,,,,,,,,,,,,#


def get_raw_transformers():
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def get_scale_transformers(size):
    return transforms.Compose([transforms.Scale((size, size)), transforms.ToTensor(),
                               transforms.Normalize(mean=[0.484, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


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
data_loader = torch.utils.data.DataLoader(data_set, batch_size=1, shuffle=False, num_workers=4)
print len(data_loader)
print data_set[0].shape


# ,,,,,,,,,,,,,,,,,,,,,,Model Class,,,,,,,,,,,,,,,,,,,,,,,,,,,#
# ,,,,,,,,,,,,,,,,,,,,,,C3D_Var_Auto,,,,,,,,,,,,,,,,#

class C3DVAE(nn.Module):
    def __init__(self):
        super(C3DVAE, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))
        self.fc6 = nn.Linear(8192, 4096)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.upsample = nn.Upsample()

        # autoencoder
        # encoder

        self.encoder = nn.Sequential(
            nn.Linear(16384, 4096),
            nn.Tanh(),
            nn.Linear(4096, 1024),
            nn.Tanh(),
            nn.Linear(1024, 256),
            nn.Tanh(),
            nn.Linear(256, 64),
        )
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.Tanh(),
            nn.Linear(256, 1024),
            nn.Tanh(),
            nn.Linear(1024, 4096),
            nn.Tanh(),
            nn.Linear(4096, 16384),
        )

        self.layer1 = nn.Sequential(
            nn.Linear(16384, 4096),
            nn.Tanh(),)

        self.layer2 = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.Tanh(),)

        self.layer3 = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.Tanh(),)
        self.layer4 = nn.Sequential(
            nn.Linear(4096, 16384),)

    # ,,,,,,,,,,, Representation Units ,,,,,,,,,,,,,,,,,,,,,,,,#

    #  auto_encoder units

    def auto_enco(self, x):
        encoded = self.encoder(x)
        return encoded

    def auto_deco(self, enco):
        decoded = self.decoder(enco)
        return decoded

    # C3D #
    def c3d(self, x):
        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)

        h = h.view(-1, 8192)
        h = self.relu(self.fc6(h))
        h = self.dropout(h)
        logits = self.fc8(h)
        return logits

    # A^_lt

    def a_lt(self, x):
        base = c3d(x)
        RE = va_auto (base)
        return RE

    # E_lt

    def e_lt(self, a_gt, a_lt):
        E_positive = self.relu(a_gt - a_lt)
        E_Negative = self.relu(a_lt - a_gt)
        return E_Negative, E_positive

    # R_lt

    def r_lt(self, e_lt_1, r_lt_1, rllt ):
        return [e_lt_1, r_lt_1, rllt]


    def forward(self, x):










model = C3DVAE()
if torch.cuda.is_available():
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_function = torch.nn.MSELoss()

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(data_loader):
        img = data
        img = img.view(-1, 4096)
        img = Variable(img)
        if torch.cuda.is_available():
            img = img.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(img)
        loss = loss_function(recon_batch, img, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % 100 == 0:
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