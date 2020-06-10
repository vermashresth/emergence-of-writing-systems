import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_classification, make_blobs

def get_dummy_data(n_samples, n_features, centers):
    X1, Y1 = make_blobs(n_samples, n_features, centers, random_state=100)
    scaler = MinMaxScaler()
    X1 = scaler.fit_transform(X1)
    return X1, Y1, X1.shape[1]

def get_data_points(csvpath, n_all_features, isNormalize=True, isPCA=True):
    df = pd.read_csv(csvpath)

    X1 = df.iloc[:, 1:n_all_features+1].values
    Y1 = df.iloc[:, -1].values

    if isPCA:
        pca = PCA(0.95)
        pca.fit(X1)
        print("PCA components", pca.n_components)
        X1 = pca.transform(X1)

    if isNormalize:
        scaler = MinMaxScaler()
        X1 = scaler.fit_transform(X1)

    print("X shape: {}, Y shape: {}".format(X1.shape, Y1.shape))
    return X1, Y1, X1.shape[1]


def get_cifar100_aggregated_data():
    coarse_label = ['apple', 'aquarium_fish','baby','bear','beaver','bed','bee','beetle','bicycle',
    'bottle','bowl','boy','bridge','bus','butterfly','camel','can','castle','caterpillar','cattle',
    'chair','chimpanzee','clock','cloud','cockroach','couch','crab','crocodile','cup','dinosaur',
    'dolphin','elephant','flatfish','forest','fox','girl','hamster','house','kangaroo','computer_keyboard',
    'lamp','lawn_mower','leopard','lion','lizard','lobster','man','maple_tree','motorcycle','mountain','mouse',
    'mushroom','oak_tree','orange','orchid','otter','palm_tree','pear','pickup_truck','pine_tree','plain','plate',
    'poppy','porcupine','possum','rabbit','raccoon','ray','road','rocket','rose','sea','seal','shark','shrew','skunk',
    'skyscraper','snail','snake','spider','squirrel','streetcar','sunflower','sweet_pepper','table','tank','telephone',
    'television','tiger','tractor','train','trout','tulip','turtle','wardrobe','whale','willow_tree','wolf',
    'woman','worm',
    ]

    mapping = {
    'aquatic mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
    'flowers': [ 'poppy', 'rose', 'sunflower', 'orchid', 'tulip'],
    'food containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
    'fruit and vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
    'household electrical device': [ 'lamp', 'telephone', 'television', 'clock', 'computer_keyboard',],
    'household furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
    'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    'large carnivores': ['bear', 'leopard', 'tiger', 'lion', 'wolf'],
    'large man-made outdoor things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
    'large natural outdoor scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
    'large omnivores and herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
    'medium-sized mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    'non-insect invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
    'people': ['baby', 'girl', 'man', 'woman', 'boy'],
    'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    'small mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
    'vehicles 1': ['bicycle', 'bus', 'train', 'motorcycle', 'pickup_truck'],
    'vehicles 2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'],
    }

    class_mapper = {i:coarse_label.index(i) for i in coarse_label}

    interest_concepts = ['fish', 'flowers', 'fruit and vegetables', 'insects', 'household furniture', 'large carnivores', 'people', 'trees', 'vehicles 1' ]

    interest_concepts = ['flowers', 'household furniture', 'large carnivores', 'people', 'vehicles 1' ]

    interest_classes = []
    for i in interest_concepts:
        for j in mapping[i][:3]:
            interest_classes.append(class_mapper[j])
    unique_concepts = list(np.unique(np.array(interest_classes)))

    new_map = {i:unique_concepts.index(i) for i in unique_concepts}

    print(len(unique_concepts))
    newdf = df[[i in interest_classes for i in df.iloc[:,-1]]]
    print(newdf.shape)
    X1 = newdf.iloc[:, 1:n_all_features+1].values
    from sklearn.decomposition import PCA
    pca = PCA(0.95)
    pca.fit(X1)
    print(pca.n_components)
    X1 = pca.transform(X1)
    print(X1.shape)
    Y1 = newdf.iloc[:, -1].values
    # X1 = keras.utils.to_categorical(y=Y1, num_classes=10)
    print(X1.shape, Y1.shape)
    # Y1 = df.iloc[:, -1].values
    # print("Successfully read data from csv")
    scaler = MinMaxScaler()
    X1 = scaler.fit_transform(X1)
    print(X1[0])
    Y1
    Ynew = []
    for i in Y1:
        Ynew.append(new_map[i])
    Y1 = np.array(Ynew)

    return X1, Y1, X1.shape
