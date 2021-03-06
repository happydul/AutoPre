import os
import torch
from torch.autograd import Variable
from Auto_enco import MyAuto
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image


torch.cuda.set_device(0)
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


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 128, 128)
    return x


def read_angle_file(filepath):
    with open(filepath, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
        return labels


def main():
    # load data
    if not os.path.exists('./nextsteering_img'):
        os.mkdir('./nextsteering_img')
    img_dir = "/home/disk/duli/My_autoencoder/data/driving_data/train"
    tran = get_scale_transformers(128)
    data_set = ImageFolderDataSets(img_dir, tran)
    total_images = data_set.__len__()
    print total_images
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print len(data_loader)
    print data_set[0].shape
    # get trained model
    auto_encoder = MyAuto()
    auto_encoder.load_state_dict(torch.load('nextsteering_autoencoder.pth'))
    auto_encoder.cuda()
    auto_encoder.eval()
    optimizer = torch.optim.Adam(auto_encoder.parameters(), lr=LR)
    loss_function = torch.nn.MSELoss()
    # Training
    for epoch in range(EPOCH):
        for step, img in enumerate(data_loader):
            b_x = Variable(img)
            b_x = b_x.view(-1, 16384)
            b_x = b_x.cuda()
            
            if step == 0:
      		E_las = torch.zeros(img.shape)
                R_pos = torch.zeros(img.shape)
    		R_neg = torch.zeros(img.shape)
                A_t = torch.zeros(img.shape)
                A_t = Variable(A_t)
                A_t = A_t.view(-1, 16384)
                A_t = A_t.cuda()
                              
            if step == 1:
		
                encoded, decoded = auto_encoder(A_t)
                loss = loss_function(decoded, b_x)
                A_t = b_x
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pic = to_img(decoded.cpu().data)
                if epoch % 10 == 0:
                    save_image(pic, './nextsteering_img/image_{}.jpg'.format(step))
   	    if step == 2:
		





        print ('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, EPOCH, loss.data[0]))
        torch.save(auto_encoder.state_dict(), './nextsteering_autoencoder.pth')
        torch.save(auto_encoder, 'nextsteering_model.pkl')

if __name__ == '__main__':
    main()
