# Day 1: Data Science and Data Manipulation using Python
This day is all about making everyone on the same page. 

Our goal for today is for you to implement "data manipulating techniques" that is far more advanced than Excel (formally called as *Data Wrangling*). 

We start off by getting familiar with a well-acclaimed programming language that is as close as writing a pseudocode called **python**

Afterwards, we start getting our hands dirty by doing your very first data science project.

I hope that after submitting all your day 1 requirements, you would hate on using Excel for your *data wrangling* needs ever again!

## A. Prerequisites:   
1. Day 0   
2. Prior Readings: Bash Script
3. Installed the ff:
    - git
    - hub
    - anaconda


Open your terminal, afterwards you should see something like this
![terminal-0](assets/terminal0.png)

If you haven't please follow the ff:

#### For MacOS, 
1. Install Xcode command line first. Just follow the prompt
```bash
$xcode-select --install
```
2. copy and paste below into your terminal to install Homebrew:
```bash
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```
2. afterwards, type the ff to install necessary packages:
```bash
$brew install git wget
```
3. Download & Install Anaconda, just follow the prompt using default setup
```bash
$wget https://repo.anaconda.com/archive/Anaconda3-2019.03-MacOSX-x86_64.sh -O ~/anaconda.sh
$bash ~/anaconda.sh
```

#### For Ubuntu/Windows Subsystem
1. Install the necessary packages, type your password if prompted:
```bash
$sudo apt-get install -y git wget
```
2. Download & Install Anaconda, just follow the prompt using default setup
```bash
$wget https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh -O ~/anaconda.sh
$bash ~/anaconda.sh
```
## B. Gettings Started

1. Go inside your repository 

2. Create the environment `deeplearning` you needed to run the program
```bash
$conda env create -n deeplearning -f environment.yml
```
3. Activate the environment
```bash
$conda activate deeplearning
```
4. Run the Jupyter lab
```bash
(deeplearning)$jupyter lab
```

## C. 10am - 1pm: Intro to Python and its Data Science Toolpack

Expected Outcome:   
- Familiarize on python   
- Implement and run your own python codes   
- work your way on your new environment   
- Familiarize with the data science tools   

### **Your Mission: Complete The Exercises**

Get your thesis group, complete at least *n* exercises where *n* is the number of group members

Here are some quicklinks: 

Data Analysis:   
Ex.1: [US_Baby Names](exercises/US_Baby_Names/Exercises.ipynb)   
Ex.2: [Online Retail](exercises/Online_Retail/Exercises.ipynb)    
Ex.3: [Apple Stock](exercises/Apple_Stock/Exercises.ipynb)   

Mastering Python:   
Ex.4: [Functions](exercises/Exercises_A.ipynb)   
Ex.5: [Data Structure](exercises/Exercises_B.ipynb)   
Ex.6: [Numpy](exercises/Exercises_C.ipynb)   
Ex.7: [Plotting](exercises/Exercises_D.ipynb)   

Always remember the creed:
> *learn on the fly*

### **Tutorials: Basic Python and Data Science**
Thus, these set of materials below are organized for you to get you more comfortable for solving the exercises.

| Basic                                                             | Intermediate                                         | Libraries                         |
| ----------------------------------------------------------------- | ---------------------------------------------------- | --------------------------------- |
| [Input and Output](basic/B1%20Input%20Output.ipynb)               | [Functions](basic/B4%20Functions.ipynb)              | [Numpy](basic/B6%20Numpy.ipynb)   |
| [Conditional Statements](basic/B2%20If%20Else.ipynb)   | [Data Structure](basic/B5%20Data%20structures.ipynb) | [Pandas](basic/B7%20Pandas.ipynb) |
| [Loop Statements](basic/B3%20Loop%20Statement.ipynb)              |                                                      |                                   |

## D. 2pm - 5pm: Your First Data Science Project
Expected Outcome:    
- Implement your own project    
- recognize the importance of Feature Selection and Engineering    
- Know the lifecycle of Machine Learning Projects   

Congratulations on surviving your morning exercise!!

Basically, at this stage, we would now focus ourselves on crafting our newly found skils.

Before we start, I just want to let you know that the most efficient way to learn is by studying a lot of examples, learning from others, and experimenting on our own. It's not about having formal studies or what. It's really about getting more hands-on as fast as possible. This is the reason why this workshop is called a crash course!!

For Data science, based on my experience, the best avenue is from the Website called [Kaggle](https://www.kaggle.com/)

So for this afternoon,    
**your mission is to submit *n* entries in a kaggle competition**    
where n is the number of members in your group

Step 1. choose one of the ff competitions:

Less Comfortable (simpler, classify into 1 or 0):    
[Titanic Dataset](https://www.kaggle.com/c/titanic)

More comfortable (harder, predict prices):   
[Housing Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

Step 2. Sign Up in Kaggle

Step 3. Browse and Study Examples.   
It's either by:
-  clicking Overview > Tutorials; or
-  Kernels > (some high rating notebooks)

Step 4. Enjoy

## E. Day 1 Homework and Day 2 Readings

Objective:  understand more concepts learned in day 1
### Getting Started

### Day 1 Homework
You have two options here.
#### Option 1: Submit to other Competition. (more comfortable, harder)
- If you chose Titanic, submit an entry to Housing Price Dataset

#### Option 2: Improve your score to the competition you chose (less comfortable, simpler)
- Get at least 1% gain in your current score, that's already okay.

#### Option 3: Finish all Day 1 Morning Exercise (simplest)
-  Just try to finish all Data Analysis Day 1 Morning Exercise

##### Additional Reference:
[Data Science Summary](https://github.com/ShuaiW/data-science-question-answer#statistics-and-ml-in-general) - I like this one as it summarizes everything you need to know

### Day 2 Readings
#### Read Chapter 1 and 2 on [Hands-on Machine Learning by Auréllien Géron](https://drive.google.com/drive/folders/15angXfk0k3Q0k_ae4VBddBRkl5cExPTS?usp=sharing)
[![geron](https://images-na.ssl-images-amazon.com/images/I/511IU--e4%2BL._SX258_BO1,204,203,200_.jpg)]((https://drive.google.com/drive/folders/15angXfk0k3Q0k_ae4VBddBRkl5cExPTS?usp=sharing))
- For convenience, you can [click here for a copy](https://drive.google.com/drive/folders/15angXfk0k3Q0k_ae4VBddBRkl5cExPTS?usp=sharing)


Note: This book has hands-on notebook you can fiddle with:   

[Chapter 1 The Machine Learning Landscape](./01_the_machine_learning_landscape.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ageron/handson-ml/blob/master/01_the_machine_learning_landscape.ipynb)

[Chapter 2 End to End Machine Learning Project](./02_end_to_end_machine_learning_project.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ageron/handson-ml/blob/master/02_end_to_end_machine_learning_project.ipynb)

Tip: You can use our `deeplearning` environment in jupyterlab to run all his codes.

#### New Feature: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb) 

1. If you haven't setup your laptop. You can click on [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb) beside each exercise so that you won't have to setup anything. Colab is a free cloud-based Jupyter notebook sponsored by Google.

2. Afterwards, Click File > Save a copy in Drive. So that you can edit and run it, and it is automatically saved in your Google Drive > My Drive > Colab Notebook

3. Lastly, Change/Verify your Runtime. Click Runtime > Change Runtime Type. It should have the ff:   
    Runtime Type: Python 3   
    Hardware Accelerator: TPU   


#### Watch ThreeBlueOneBrown in youtube, four episodes
- Click on the images/links below so that you can watch it.

[Episode 1: What is a neural Network](https://youtu.be/aircAruvnKk)    
[![](http://img.youtube.com/vi/aircAruvnKk/0.jpg)](http://www.youtube.com/watch?v=aircAruvnKk "")

[Episode 2: Gradient Descent](https://youtu.be/IHZwWFHWa-w)    
[![](http://img.youtube.com/vi/IHZwWFHWa-w/0.jpg)](http://www.youtube.com/watch?v=IHZwWFHWa-w "")

[Episode 3: Backpropagation](https://youtu.be/Ilg3gGewQ5U)   
[![](http://img.youtube.com/vi/Ilg3gGewQ5U/0.jpg)](http://www.youtube.com/watch?v=Ilg3gGewQ5U "")    

[Episode 4: Backpropagation Calculus](https://youtu.be/tIeHLnjs5U8)    
[![](http://img.youtube.com/vi/tIeHLnjs5U8/0.jpg)](http://www.youtube.com/watch?v=tIeHLnjs5U8 "")    
