import torch
import torchvision.datasets as datasets
from tqdm import tqdm

class InferLoader(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super(InferLoader, self).__init__(root, transform=transform, target_transform=target_transform)
        self.id = self.samples
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target, path.split('/')[-1].split('.')[0]
    
def evaluate(model, dataloader, device):
    with open('solution.csv', 'w') as f:
        f.write('id,affenpinscher,afghan_hound,african_hunting_dog,airedale,american_staffordshire_terrier,appenzeller,australian_terrier,basenji,basset,beagle,bedlington_terrier,bernese_mountain_dog,black_and_tan_coonhound,blenheim_spaniel,bloodhound,bluetick,border_collie,border_terrier,borzoi,boston_bull,bouvier_des_flandres,boxer,brabancon_griffon,briard,brittany_spaniel,bull_mastiff,cairn,cardigan,chesapeake_bay_retriever,chihuahua,chow,clumber,cocker_spaniel,collie,curly_coated_retriever,dandie_dinmont,dhole,dingo,doberman,english_foxhound,english_setter,english_springer,entlebucher,eskimo_dog,flat_coated_retriever,french_bulldog,german_shepherd,german_short_haired_pointer,giant_schnauzer,golden_retriever,gordon_setter,great_dane,great_pyrenees,greater_swiss_mountain_dog,groenendael,ibizan_hound,irish_setter,irish_terrier,irish_water_spaniel,irish_wolfhound,italian_greyhound,japanese_spaniel,keeshond,kelpie,kerry_blue_terrier,komondor,kuvasz,labrador_retriever,lakeland_terrier,leonberg,lhasa,malamute,malinois,maltese_dog,mexican_hairless,miniature_pinscher,miniature_poodle,miniature_schnauzer,newfoundland,norfolk_terrier,norwegian_elkhound,norwich_terrier,old_english_sheepdog,otterhound,papillon,pekinese,pembroke,pomeranian,pug,redbone,rhodesian_ridgeback,rottweiler,saint_bernard,saluki,samoyed,schipperke,scotch_terrier,scottish_deerhound,sealyham_terrier,shetland_sheepdog,shih_tzu,siberian_husky,silky_terrier,soft_coated_wheaten_terrier,staffordshire_bullterrier,standard_poodle,standard_schnauzer,sussex_spaniel,tibetan_mastiff,tibetan_terrier,toy_poodle,toy_terrier,vizsla,walker_hound,weimaraner,welsh_springer_spaniel,west_highland_white_terrier,whippet,wire_haired_fox_terrier,yorkshire_terrier')
        for data in tqdm(dataloader, desc='Evaluating'):
            images, id, img_name_list = data
            images = images.to(device)
            output = model(images)
            output = torch.softmax(output, dim=1)
            for i in range(output.size(0)):
                img_name = img_name_list[i]
                f.write('\n' + img_name)
                for j in range(output.size(1)):
                    f.write(',%.9f' % output[i][j].item())