def augment(self,clip,transformations = None):
        new_clips = []
        if transformations is None:
            transformations = random.sample(self.transforms_list,random.randint(1,8))+[RandomCrop(224,pad_if_needed = True)]
            print(transformations)
            #print(transformations.transforms)
            transforms_list = []
            for t in transformations:
                #print(t)
                transforms_list.append(t.get_transform())
            transformations = transforms.Compose(transforms_list)
        for fn in range(clip.shape[0]):
            new_clips.append(transformations(clip[fn]).unsqueeze(0))
