import os
from random import gauss, randint, choice, sample, seed
from scipy.io import loadmat
import torch
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.datasets import DatasetFolder
from torchvision.transforms import Resize, CenterCrop, RandomCrop, RandomResizedCrop, RandomRotation, \
    ColorJitter, RandomErasing, RandomHorizontalFlip, ToTensor, ToPILImage, Normalize, Grayscale, Compose

all_classes = ['BEST', 'RDS', 'TTN']
# basePath = 'drive/My Drive/DatasetClassificazione'
# basePath_avi = './Video_LUStory_Ingegneria/'
basePath_avi = '/Volumes/SD Card/ICPR/Dataset_RAW'
# basePath_mat = './Video_LUStory_Ingegneria_mat/'
basePath_mat = '/Volumes/SD Card/ICPR/Dataset_processato/Dataset_f'

CV_FOLD = {'BEST': [[1, 2],
                    [3, 4, 21],
                    [5, 6],
                    [7, 8],
                    [9, 10],
                    [11, 12, 22],
                    [13, 14],
                    [15, 16],
                    [17, 18],
                    [19, 20, 23]],
           'RDS': [[24, 33, 51],
                   [25, 38],
                   [26, 35, 49],
                   [34, 44, 52],
                   [27, 40, 45],
                   [30, 43, 47],
                   [28, 36, 39],
                   [29, 37, 50],
                   [31, 41, 46],
                   [32, 42, 48]],
           'TTN': [[54, 64],
                   [55],
                   [56],
                   [57],
                   [58, 65],
                   [59],
                   [60],
                   [61],
                   [62, 66],
                   [63]]}

# train_transform = lambda num_rows: Compose([ToPILImage(), RandomCrop(num_rows), RandomHorizontalFlip(), ToTensor()])
# train_transform = lambda num_rows: Compose([ToPILImage(), GaussianRandomCrop(num_rows, 50), RandomHorizontalFlip(), ToTensor()])
# test_transform = lambda num_rows: Compose([ToPILImage(), CenterCrop(num_rows), ToTensor()])
# train_transform = lambda num_rows: Compose([ToPILImage(),
#                                             Resize((num_rows, 461)),
#                                             RandomHorizontalFlip(), ToTensor(),
#                                             # Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#                                             Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                                             ])
train_transform = lambda num_rows: Compose([ToPILImage(),
                                            # Resize((num_rows, 461)),
                                            # Resize((num_rows+10, 461)), RandomCrop((num_rows, 461)),
                                            RandomResizedCrop((num_rows, 461), scale=(0.99, 1.0), ratio=(0.99, 1.01)),
                                            RandomRotation(10),
                                            ColorJitter(.15, .15),
                                            # ColorJitter(.25, .25),
                                            RandomHorizontalFlip(), ToTensor(),
                                            # Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                            RandomErasing(p=.5, scale=(0.02, 0.20), ratio=(0.3, 3.3), value=-1),
                                            ])
test_transform = lambda num_rows: lambda x, T=Compose([ToPILImage(), Resize((num_rows, 460)), ToTensor()]): [T(_) for _ in x]


def default_mat_loader(path, num_rows, first_frame=None, last_frame=None, all_frames=False, return_value=False):
    # definisce il loader che carica il file .mat, seleziona una frame random e prende le prime num_rows righe

    if all_frames:  # carica tutte le frame e le restituisce in una lista
        data = loadmat(path)
        if return_value:
            valore = data['valore']
        data = [data[k][:num_rows] for k in data.keys() if k.startswith('f') and len(k) < 3]

    elif first_frame is None and last_frame is None:  # carica l'intero video poi seleziona la frame random
        data = loadmat(path)
        if return_value:
            valore = data['valore']
        f = choice([k for k in data.keys() if k.startswith('f') and len(k) < 3])
        data = data[f][:num_rows]

    else:  # carica solo la frame selezionata random nell'intervallo (first_frame, last_frame)
        f = randint(first_frame, last_frame)
        if return_value:
            data = loadmat(path, variable_names=[f, 'valore'])
            valore = data['valore']
            data = data[f][:num_rows]
        else:
            data = loadmat(path, variable_names=[f])[f][:num_rows]

    if return_value:
        return data, float(valore.item()/480.)
    else:
        return data


class GaussianRandomCrop(RandomCrop):  # eredita da RandomCrop (distribuzione uniforme)
    def __init__(self, size, sigma, mu=(0, 0), padding=None, pad_if_needed=False, fill=0, padding_mode='constant', verbose=False):
        '''
        :param size: see RandomCrop
        :param sigma: (h_sigma, w_sigma) of the Gaussian distribution
        :param mu: (h_mu, w_mu) of the Gaussian distribution with respect to the center of the image
        :param padding: see RandomCrop
        :param pad_if_needed: see RandomCrop
        :param fill: see RandomCrop
        :param padding_mode: see RandomCrop
        '''
        self.mu = mu if not isinstance(mu, (int, float)) and len(mu) == 2 else (mu, mu)
        self.sigma = sigma if not isinstance(sigma, (int, float)) and len(sigma) == 2 else (sigma, sigma)
        self.verbose = verbose
        super(GaussianRandomCrop, self).__init__(size, padding, pad_if_needed, fill, padding_mode)

    def get_params(self, img, output_size):  # sostituisce la distribuzione uniforme con quella Gaussiana
        """Get parameters for ``crop`` for a Gaussian random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for Gaussian random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = round(min(max(0, gauss(self.mu[0] + (h - th)//2, self.sigma[0])), h - th))
        j = round(min(max(0, gauss(self.mu[1] + (w - tw)//2, self.sigma[1])), w - tw))
        if self.verbose:
            print('Gauss h', i, ':', i+th, 'in', h)
            print('Gauss w', j, ':', j+tw, 'in', w)
        return i, j, th, tw


def balance_datasets(datasets):
    # bilancia il numero di campioni nei vari dataset per replicazione
    l = max([len(dataset) for dataset in datasets])
    for dataset in datasets:
        while len(dataset) < l:
            dataset.samples += sample(dataset.samples, min(len(dataset), l - len(dataset)))
    return datasets


def replicate_datasets(datasets, increase_factor=0):
    # aumenta il numero di campioni di tutti i datset per replicazione per far durare di più l'epoca
    if increase_factor > 1:
        for dataset in datasets:
            dataset.samples = dataset.samples * increase_factor
    return datasets


class BalanceConcatDataset(ConcatDataset):  # eredita da ConcatDataset
    def __init__(self, datasets):
        datasets = balance_datasets(datasets)
        super(BalanceConcatDataset, self).__init__(datasets)


class LUSFolder(DatasetFolder):  # eredita da DatasetFolder
    def __init__(self, root, train_phase, target_value=False, subset_in=None, subset_out=None, num_rows=224,
                 subset_var='paziente', exclude_class=None, exclude_val_higher=None, random_seed=0, loader=None):
        seed(random_seed)
        self.target_value = target_value

        # definisce le trasformazioni da effettuare
        transform = train_transform(num_rows) if train_phase else test_transform(num_rows)
        if loader is None:
            if train_phase:
                loader = lambda path, first_frame=None, last_frame=None: default_mat_loader(path, num_rows, first_frame, last_frame, return_value=target_value)
            else:
                loader = lambda path, first_frame=None, last_frame=None: default_mat_loader(path, num_rows, all_frames=True, return_value=target_value)
        # definisce la funzione che seleziona i file in base a subset_in e subset_out
        assert subset_in is None or subset_out is None  # almeno uno dei due deve essere None
        if subset_in is not None:
            # prende i file dei soggetti presenti in subset_in
            if exclude_val_higher is None:
                is_valid_fun = lambda path: path.endswith('.mat') and (exclude_class is None or all([e not in path for e in exclude_class])) \
                                            and loadmat(path, variable_names=subset_var)[subset_var] in subset_in
            else:
                is_valid_fun = lambda path: path.endswith('.mat') and (exclude_class is None or all([e not in path for e in exclude_class])) \
                                            and loadmat(path, variable_names=subset_var)[subset_var] in subset_in \
                                            and loadmat(path, variable_names='valore')['valore'] < exclude_val_higher
        elif subset_out is not None:
            # prende i file dei soggetti non presenti in subset_in
            if exclude_val_higher is None:
                is_valid_fun = lambda path: path.endswith('.mat') and (exclude_class is None or all([e not in path for e in exclude_class])) \
                                            and loadmat(path, variable_names=subset_var)[subset_var] not in subset_out
            else:
                is_valid_fun = lambda path: path.endswith('.mat') and (exclude_class is None or all([e not in path for e in exclude_class])) \
                                            and loadmat(path, variable_names=subset_var)[subset_var] not in subset_out \
                                            and loadmat(path, variable_names='valore')['valore'] < exclude_val_higher
        else:
            # prende tutti i file
            if exclude_val_higher is None:
                is_valid_fun = lambda path: path.endswith('.mat') and (exclude_class is None or all([e not in path for e in exclude_class]))
            else:
                is_valid_fun = lambda path: path.endswith('.mat') and (exclude_class is None or all([e not in path for e in exclude_class])) \
                                            and loadmat(path, variable_names='valore')['valore'] < exclude_val_higher

        # costruttore
        super(LUSFolder, self).__init__(root=root, loader=loader, extensions=None,
                                        transform=transform, target_transform=None, is_valid_file=is_valid_fun)
        self.imgs = self.samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        if self.target_value:  # loader restituisce anche il valore target (sovrascritto alla label della classe)
            sample, target = self.loader(path)
        else:
            sample = self.loader(path)
        if self.transform is not None:
            #sample in shape: 224, 461, 3
            sample = self.transform(sample)
            #sample out shape: 3, 224, 461
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class LUSList(DatasetFolder):  # eredita da DatasetFolder
    def __init__(self, filelist, train_phase, subset_in=None, subset_out=None, num_rows=224,
                 subset_var='paziente', exclude_class=None, random_seed=0, filelist_separator=',', loader=default_mat_loader):
        seed(random_seed)

        self.filelist = filelist
        self.num_rows = num_rows

        # definisce le trasformazioni da effettuare
        transform = train_transform(num_rows) if train_phase else test_transform(num_rows)

        # definisce la funzione che seleziona i file in base a subset_in e subset_out
        assert subset_in is None or subset_out is None  # almeno uno dei due deve essere None
        check_path = lambda path: os.path.isfile(path) and path.endswith('.mat') and (exclude_class is None or all([e not in path for e in exclude_class]))
        if subset_in is not None:
            # prende i file dei soggetti presenti in subset_in
            is_valid_fun = lambda path: check_path(path) and loadmat(path)[subset_var] in subset_in
        elif subset_out is not None:
            # prende i file dei soggetti non presenti in subset_in
            is_valid_fun = lambda path: check_path(path) and loadmat(path)[subset_var] not in subset_out
        else:
            # prende tutti i file
            is_valid_fun = check_path

        # costruttore
        with open(filelist, 'r') as file:
            samples = [line.split(filelist_separator) for line in file.readlines()]
        # scarta i path non validi
        samples = [e for e in samples if is_valid_fun(e[0])]
        if len(samples) == 0:
            raise (RuntimeError("Found 0 existing files in list: " + self.filelist))

        # self.classes = classes
        # self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.imgs = self.samples
        self.loader = lambda path, first_frame=None, last_frame=None: loader(path, num_rows, first_frame, last_frame)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target, first_frame, last_frame = self.samples[index]
        sample = self.loader(path, first_frame, last_frame)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


def test_Gaussian_random_crop(img_sz=100, crop_sz=10, sig=(0, 5)):
    class MyRandomCrop(RandomCrop):
        @staticmethod
        def get_params(img, output_size):
            w, h = img.size
            th, tw = output_size
            if w == tw and h == th:
                return 0, 0, h, w

            i = randint(0, h - th)
            j = randint(0, w - tw)
            print('\nUnif  h', i, ':', i + th, 'in', h)
            print('Unif  w', j, ':', j + tw, 'in', w)
            return i, j, th, tw

    import numpy as np

    tu = Compose([ToPILImage(), MyRandomCrop(crop_sz)])
    # tg = Compose([ToPILImage(), GaussianRandomCrop(crop_sz, sig, verbose=True)])
    tg = Compose([ToPILImage(), GaussianRandomCrop(crop_sz, sig, (-img_sz // 2 + crop_sz // 2, 0), verbose=True)])
    for i in range(100):
        tu(np.zeros((img_sz, img_sz, 3), np.uint8))
        tg(np.zeros((img_sz, img_sz, 3), np.uint8))


def show_images(data_loader, batches=10):
    # salvataggio immagini per controllo
    from os import makedirs
    from shutil import rmtree
    # save_img = lambda tensor, name: ToPILImage()((tensor.numpy().transpose((1, 2, 0)).squeeze() * 255.)  # w/o Normalization
    #                                              .astype('uint8')).save(name)
    save_img = lambda tensor, name: ToPILImage()((tensor.numpy().transpose((1, 2, 0)).squeeze() * 128. + 128.)  # w/ Normalization
                                                 .clip(0, 255).astype('uint8')).save(name)
    try: rmtree('./temp')
    except: pass
    makedirs('./temp', exist_ok=True)
    for b, (X, y) in enumerate(data_loader):
        for i, x in enumerate(X):
            save_img(x, 'temp/batch%d_img%d_c%d.png' % (b, i, y[i].item()))
        print('batch', b, 'written')
        if b >= batches:
            break


def get_mat_dataloaders(classes, target_value=False, replicate_minority_classes=True, fold_test=0, fold_val=None, batch_size=32, num_workers=4, replicate_all_classes=10):
    if fold_val is None: fold_val = fold_test - 1
    print('Validation fold:', fold_val, '\nTest fold:', fold_test)
    if len(classes) == 2 and all_classes[-1] in classes:
        Warning('Correggere le label del dataset!')
    basePath = basePath_mat

    # creazione dataset per classe con selezione degli utenti
    train_ds, val_ds, test_ds = [], [], []
    for class_name in classes:
        print('- creating data sets for class', class_name)
        exclude_class = all_classes.copy()
        exclude_class.remove(class_name)
        train_ds.append(LUSFolder(root=basePath, train_phase=True, target_value=target_value, subset_out=CV_FOLD[class_name][fold_test] + CV_FOLD[class_name][fold_val],
                                  exclude_class=exclude_class, exclude_val_higher=450 if not target_value and class_name is not 'BEST' else None))
        val_ds.append(LUSFolder(root=basePath, train_phase=False, target_value=target_value, subset_in=CV_FOLD[class_name][fold_val],
                                exclude_class=exclude_class, exclude_val_higher=450 if not target_value and class_name is not 'BEST' else None))
        test_ds.append(LUSFolder(root=basePath, train_phase=False, target_value=target_value, subset_in=CV_FOLD[class_name][fold_test],
                                 exclude_class=exclude_class, exclude_val_higher=450 if not target_value and class_name is not 'BEST' else None))
        print('  - found train/val/test samples:', len(train_ds[-1]), len(val_ds[-1]), len(test_ds[-1]))

    # bilanciamento del numero di campioni delle classi ed eventuale replicazione di tutti i campioni
    if replicate_minority_classes:
        print('- balancing data sets by sample duplications (replicate_all_classes=%d)' % replicate_all_classes)
        print('  - before:', [len(_) for _ in train_ds])
        train_ds = balance_datasets(train_ds)
        print('  - after:', [len(_) for _ in train_ds])
    if replicate_all_classes > 1:
        print('- replicating data sets by sample duplications %d times' % replicate_all_classes)
        print('  - before:', [len(_) for _ in train_ds])
        train_ds = replicate_datasets(train_ds, replicate_all_classes)
        print('  - after:', [len(_) for _ in train_ds])

    # data loader sulla concatenazione dei dataset delle singole classi
    print('- creating data loaders')
    def collate_fn(data):  # collate all frames of a validation or test video
                            # data: list of batch_size//5 tuple (batch). Each tuple contains a list of the img frames (as tensors) and the label
        X, Y = [], []
        for x, y in data:
            X += x
            Y += [y] * len(x)
        return torch.stack(X), torch.Tensor(Y)
    train_dl = DataLoader(ConcatDataset(train_ds), num_workers=num_workers, pin_memory=True,
                          shuffle=True, batch_size=batch_size)
    val_dl = DataLoader(ConcatDataset(val_ds), num_workers=num_workers, pin_memory=True,
                        shuffle=False, batch_size=batch_size//5, collate_fn=collate_fn)
    test_dl = DataLoader(ConcatDataset(test_ds), num_workers=num_workers, pin_memory=True,
                         shuffle=False, batch_size=batch_size//5, collate_fn=collate_fn)
    return train_dl, val_dl, test_dl


if __name__ == "__main__":
    # classificazione_ternaria()

    train_dl, val_dl, test_dl = get_mat_dataloaders(all_classes, num_workers=0)
    # enumerate(train_dl)
    # show_images(train_dl)

    X_train,y_train = next(iter(train_dl))    
    X_test,y_test = next(iter(test_dl))
    X_val,y_val = next(iter(val_dl))
    
#%% image showing
    import matplotlib.pyplot as plt
    import numpy as np
    
    # iterator = iter(train_dl)
    X,y = next(iter(train_dl))
    
    img_transform = train_transform(224)(img)
    
    img = X[10], img_transform = X_transform[10]
    img_t = np.transpose(img, (1,2,0))
    img_transform_t = np.transpose(img_transform, (1,2,0))
    plt.imshow(img_t, cmap='gray')
    plt.imshow(img_transform_t, cmap='gray')



#%% test_dl

    X,y = next(iter(test_dl))
    X_val,y_val = next(iter(val_dl))

#%% Transformers

    def revalue_image(image):
        return (image - image.min()) / (image.max() - image.min())
    
    
    def adapt_imageTensor_toPlot(imageTensor):
        imageTensor_t = torch.transpose(imageTensor, 0,1)
        return torch.transpose(imageTensor_t, 1,2)
    
    from PIL import Image
    import matplotlib.pyplot as plt
    image = np.asarray(Image.open('trial_image.png'),).astype('float').transpose(2,0,1)
    image_tensor = torch.from_numpy(image)
    image_tensor_revalued = revalue_image(image_tensor)
    image_tensor_revalued_transformed = train_transform(224)(image_tensor_revalued)
    
    
    plt.imshow(image)
    # plt.show()
    plt.imshow(image_tensor)
    plt.imshow(adapt_imageTensor_toPlot(image_tensor_revalued))
    plt.imshow(adapt_imageTensor_toPlot(image_tensor_revalued_transformed))
    # plt.show()



#%% TEST: randomResizedCrop

    from PIL import Image
    x_img = torch.from_numpy(np.asarray(Image.open('trial_image.png'),).astype('int').transpose(2,0,1))
    x_img_transformed_1 = RandomResizedCrop((512, 512), scale=(0.5,0.5), ratio=(1.0,1.0))(x_image)
    fig, axs = plt.subplots(2,1); fig.suptitle('RandomResizedCrop') 
    axs[0].imshow(x_img.permute(1,2,0))
    axs[1].imshow(x_img_transformed_1.permute(1,2,0)/255.0)
    
    x_img_transformed_2 = RandomRotation(10)(x_img_transformed_1)
    fig, axs = plt.subplots(2,1); fig.suptitle('RandomRotation') 
    axs[0].imshow(x_img_transformed_1.permute(1,2,0)/255.0)
    axs[1].imshow(x_img_transformed_2.permute(1,2,0)/255.0)
    
    x_img_transformed_3 = ColorJitter(.15, .15)(x_img_transformed_2)
    fig, axs = plt.subplots(2,1); fig.suptitle('ColorJitter')
    axs[0].imshow(x_img_transformed_2.permute(1,2,0)/255.0)
    axs[1].imshow(x_img_transformed_3.permute(1,2,0))
    
    x_img_transformed_4 = RandomHorizontalFlip()(x_img_transformed_)
    fig, axs = plt.subplots(2,1); fig.suptitle('RandomHorizontalFlip')
    axs[0].imshow(x_img_transformed_3.permute(1,2,0)/255.0)
    axs[1].imshow(x_img_transformed_4.permute(1,2,0)/255.0)
    
    
    x_img_transformed_5 = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(x_img_transformed_4)
    fig, axs = plt.subplots(2,1); fig.suptitle('Normalize')
    axs[0].imshow(x_img_transformed_4.permute(1,2,0)/255.0)
    axs[1].imshow(x_img_transformed_5.permute(1,2,0)/255.0)
    
    x_img_transformed_6 = RandomErasing(p=.5, scale=(0.02, 0.20), ratio=(0.3, 3.3), value=-1)(x_img_transformed_5)
    fig, axs = plt.subplots(2,1); fig.suptitle('RandomErasing')
    axs[0].imshow(x_img_transformed_5.permute(1,2,0)/255.0)
    axs[1].imshow(x_img_transformed_6.permute(1,2,0)/255.0)







