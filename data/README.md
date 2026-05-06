
# Data Collection Process
- This dataset consists of image data focused on Indian mammal species, curated specifically for image classification tasks.

### Dataset Categories
The dataset is organized into folders by species name. Popular indian mammals are considered for this assignment. Aroung 4500 images are there accross 33 categories. It includes below categories:  

- Asiatic Lion
- Asiatic Wildcat
- Bengal Fox
- Chinkara
- Chital
- Four Horned Antelope
- Golden Jackal
- Honey Badger
- Indian Leopard
- Jungle Cat
- Nilgai
- Ruddy Mongoose
- Striped Hyena
- Wild Boar
- barasingha
- barking_deer
- bengal_tiger
- blackbuck
- bonnet_macaque
- desert_cat
- dhole
- ganges_river_dolphin
- gaur
- golden_langur
- greater_bandicoot_rat
- hanuman_langur
- himalayan_black_bear
- indian_elephant
- indian_flying_fox
- indian_giant_squirrel
- indian_grey_mongoose
- indian_hedgehog
- indian_palm_squirrel

### Data Sources

The images were sourced from the following open-access platforms:

- iNaturalist: Research-grade observations of Indian fauna

- Some images taken from Kaggle: https://www.kaggle.com/datasets/asaniczka/ mammals-image-classification-dataset-45-animals

- Google: Direct google image download

### Challenges and Solutions
- Intra-class Variation: Mammals often appear in diverse lighting, weather conditions, and camouflage. Solution: Included images with varied backgrounds and angles to improve model generalization.

- Class Imbalance: Common species had significantly more data than rare ones. Solution: Performed targeted searches for rare species and applied oversampling techniques.

- Data Noise: Initial downloads contained only background images or maps. Solution: Manual verification to remove irrelevant images.