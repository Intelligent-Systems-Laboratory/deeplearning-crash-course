# Deep Learning  Crash Course

This crash course believes in three philosophies:

The first one is:
> We learn by doing

Thus, we emphasize this philosophy by hopefully doing everything as hands-on as possible.

The second one is: 
> We are our best teachers

This means that we learn best when we learn it on our own. Based on my twenty six years of existence, the best learners I have met learns by exploring it on their own. They pursue knowledge for knowledge sake. 

Hence, in this workshop, what we want to emphasize is that the materials should be exploratory on its own. You are actually encouraged to use more reference material other than the ones contained here. 

In corollary to this, as most of the best materials are in text format. This course focuses on you becoming a reader. 

The third one is:
> We will never be ready

This means that you don't need so much pre-requisites to learn something new. 

Thus, the following are not a valid excuse:  
- *"Oh! I never learn python, so I can't do the exercise."*
- *"I am not good in programming, so this is too fast. I can't do it. "*
- *"I can't do this. How can I do it without having formal lectures?"*

The point here is as long as you can do three things:      
(1) Use your laptop for reading and writing,    
(2) Use your mouse/trackpad for navigating on your screen   
(3) Type something using your keyboard,    

then you can do it!

Last but not the least, I just want to pre-empt you that you might feel very uncomfortable as there's so many complicated things to learn in one sitting. 

So, I just want to let you know:   
**"Welcome to the world of software development... where every thing is complicated!"**

**Carpe Diem!!**

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
1. copy and paste below into your terminal to install Homebrew:
```bash
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```
2. afterwards, type the ff to install necessary packages:
```bash
$brew install git wget hub
```
3. Download Anaconda
```bash
$wget https://repo.anaconda.com/archive/Anaconda3-2019.03-MacOSX-x86_64.sh -O ~/anaconda.sh
```
4. Install Anaconda and follow the prompt
```bash
$bash ~/anaconda.sh
```

#### For Ubuntu/Windows Subsystem
1. Install the necessary packages, type your password if prompted:
```bash
$sudo apt install -y git wget hub
```
2. Download Anaconda
```bash
$wget https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh -O ~/anaconda.sh
```
3. Install Anaconda and follow the prompt
```bash
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
| [Conditional Statements](basic/02%20Control%20statements.ipynb)   | [Data Structure](basic/B5%20Data%20structures.ipynb) | [Pandas](basic/B7%20Pandas.ipynb) |
| [Loop Statements](basic/B3%20Loop%20Statement.ipynb)              |                                                      |                                   |

