STA 663 Syllabus (Spring 2016)
====

Learning Objectives
----
- Practice writing code for analysis that is reproducible

> Use of git, Jupyter, testing and role of automation

- Achieve fluency in Python

> Write idiomatic Python 3 code. There will be ample programming exercises for you to develop this skill.

- Become familiar with the most useful Python packages for solving analysis problems

> Such packages include numpy, scipy, matplotlib, pandas, scikit-learn, statsmodels, pymc3, pystan, pyspark and others

- Develop intuition for concepts (e.g. geometry) underlying statistical algorithms

> How do optimization routines, EM and MCMC actually do their magic?

- Learn how to develop statistical algorithms in Python

> Final class project requires you to develop, optimize, test and apply a statistical algorithm from the research literature

- Profile and optimize code to make good use of available resources (e.g parallel environments)

> From interpreted to compiled code (Python to C), multi-core parallelism, GPU programming and distributed computing for big data

Instructors
----
- Cliburn Chan <cliburn.chan@duke.edu>
- Janice McCarthy <janice.mccarthy@duke.edu>
- Christine Chai (TA) <christine.chai@duke.edu>
- Yuhao Liang (TA) <yuhao.liang@stat.duke.edu>

Office Hours
----

- Cliburn Chan: Thursday 6pm to 7pm Old Chem 116
- Christine Chai (TA): Monday 1pm to 3 pm Old Chem 211A
- Yuhao Liang (TA): Tuesday 7pm to 9pm Old Chem 211A

Requirements for Class
----
Please bring a laptop for each class as you will often be expected to type along. You are expected to provide your own laptop.

Grading
----
- Homework assignments 50%

> There will be up to 10 homework assignments, each of equal weight. These will typically require significant programming effort.

- Mid-term exams 25%

> This will be an in-class series of programming challenges. If you have been working hard on homework assignments these challenges should be well within your ability.

- Final project 25%

> For the final project, you will implement, test, optimize and apply a statistical algorithm from a research paper. You will submit a Github repository containing all documentation and code created. There will also be a class presentation of the project.

### Final grade calculations
- A = 90 - 100
- B = 70 - 89
- C = 50 - 69
- D = Below 50

Fractional final scores will be rounded UP.

Homework assignments
----
Homework will be assigned on Thursdays and due the following Wednesday before 4 PM. Late homework will not be accepted (automatic 0) as solutions will generally be discussed during the Wednesday lab session.

Instructions for how to submit homework will be provided shortly.

Resources
----
All notebooks, data sets and homework assignment will be posted to the GitHub repository at https://github.com/cliburn/sta-663-2016

A searchable web-accessible version of the notebooks is at http://people.duke.edu/~ccc14/sta-663-2016/

Honor Code
----
Please follow the Duke honor code. All work submitted should be from your individual effort unless given explicit instructions otherwise.

Lecture Sequence (subject to revision)
----
### Lecture 1: Introduction to Course

- Using Jupyter in Docker
- Setting up account on AWS (Amazon Web Services) cloud computing platform
- Using `git`: Cloning, pulling, pushing
- Jupyter: Literate and multi-language programming
- Python: Data structures and control flow

#### Objectives
- Log into Duke VM
- Create AWS account and VM
- Install packages with `conda`
- Set up virtual environment with `conda`
- Learn to use Jupyter notebook for markdown and
- Using `R` in Jupyter
- Use Python as an interactive calculator

### Lecture 2: Basics of Python

- Python: Functions
- Python: Text
- Python: I/O
- `string`, `re`, `itertools`, `functools`, `requests`

#### Objectives
- Write custom functions in Python
- Functional programming building blocks - map, filter, reduce
- Load and save data from files and URLs
- Basic string data munging

### Lecture 3: Working with numeric data in `numpy`

- Python: Numbers
- `numpy`, `scipy`

#### Objectives
- Create and manipulate vectors and matrices in Python

### Lecture 4: Data analysis with `pandas`

- Python: Data
- Python: Databases
- Python: Graphics
- SQL, `sqlite3`, `matpltolib`, `seaborn`

#### Objectives
- Getting data into DataFrames
- Selection of data in DataFrames
- Data summaries and cleaning
- Split-apply-combine
- Basic use of SQL to query relational databases
- Creating and customizing plots in Python

### Lecture 5: Machine learning with `scikit-learn`

- Preprocessing
- Dimension reduction
- Clustering
- Supervised learning
- Probabilistic (generative) methods
- Analysis pipelines
- Validation

#### Objectives

- Exposure to machine learning examples before we dive into the underlying algorithmic ideas
- Ability to run standard analysis on small to moderate data sets

### Lecture 6: Numerical methods

- Computer arithmetic
- Linear algebra 1
- `numpy.linalg` and `scipy.linalg`

#### Objectives
- Appreciate what floating point numbers are
- See catastrophic cancellation
- Understand basic concepts of linear algebra
- Use `linalg` library to do do linear algebra routines

### Lecture 7: Numerical methods

- Linear algebra 2
- `scipy.blas` and `scipy.lapack`

#### Objectives
- Understand matrix decomposition algorithms
- Use `linalg` to solve linear algebra problems

### Lecture 8: Numerical methods

- Theory: PCA, SVD and LSA
- `scikit-learn` and `gensim`

#### Objectives
- Understand PCA and related algorithms
- Apply PCA for dimension reduction in topic modeling

### Lecture 9: Numerical methods

- Theory: Root finding and optimization
- `numpy` and `scipy.optimize`

#### Objectives
- Statistical problems as maximizing log likelihood
- Understand Newton method in 1D
- Understand relationship between root-finidng and optimization

### Lecture 10: Numerical methods

- Theory: Multivariate optimization 1
- `scipy.optimize` and `scikit-learn`

#### Objectives
- Intuition for the Jacobian
- Understand gradient descent and stochastic gradient descent
- Apply gradient descent to solve a regression problem

### Lecture 11: Numerical methods

- Theory: Multivariate optimization 2
- `statsmodels`

#### Objectives
- Intuition for the Hessian, Fisher and observed information
- Understand Newton and conjugate gradient methods
- Understand how Newton method is used in IRLS

### Lecture 12: Numerical methods

- Theory: Expectation-Maximization 1
- `numpy`

#### Objectives
- Understand EM and data augmentations
- Apply EM to simple Bernoulli/Binomal models

### Lecture 13: Numerical methods

- Theory: Expectation-Maximization 2
- `pymix`

#### Objectives
- Deeper understanding of EM
- Applying to mixture of Gaussians
- EM for the MAP

### Lecture 14: Probabilistic methods

- Probability and random number generation
- `numpy.random` and `scipy.stats`

#### Objectives
- Intuition for how random number generators work
- The CDF and inverse transform method for generating random numbers
- Using discrete and continuous distributions

### Lecture 15: Probabilistic methods

- Simulation and resampling
- Monte Carlo methods
- `numpy.random` and `scipy.stats`

#### Objectives
- Use of simulations for point and interval estimates
- Code simple machine learning and cross-validation example
- Using Monte Carlo methods for estimating integrals

### Lecture 16: Probabilistic methods

- MCMC 1: Gibbs and Metropolis
- `numpy.random` and `scipy.stats`

#### Objectives
- Hand-coding of Metropolis sampler
- Hadn-coding of Gibbs sampler
- Inference and posterior predictive checks

### Lecture 17: Probabilistic methods

- MCMC 2: Slice and HMC
- `pymc3` and `pystan`

#### Objectives
- Intuition for slice and Hamiltonian samplers
- Use of MCMC packages to fit hierarchical models

### Lecture 18: Improving performance

- Algorithmic complexity
- Benchmarking and profiling
- Code optimization
- Using appropriate data structures
- Classic algorithmic approaches to speed up code

#### Objectives
- Intuition for performance of algorithms and data structures
- Use of Big O notation
- Benchmarking and profiling

### Lecture 19: Improving performance

- Working with `cython` and `numba`
- How to go from interpreted to compiled code using `cython`
- Using `numba` to compile code - `jit`, `njit`, `vectorize`
- Using `numba` on the GPU

#### Objectives
- Get speed-ups with compiled code

### Lecture 20: Improving performance

- Introduction to parallel programming
- Benefits of functional approach
- Synchronous and asynchronous programs
- Embarrassingly parallel programs
- Master-worker paradigm
- IPython. Parallel, `dask`, and `multiprocessing`

#### Objectives
- Understand common parallel idioms
- Run embarrassingly parallel programs on multiple cores
- Run shared memory programs on multiple cores

### Lecture 21: Distributed computing

- Working with massive data sets
- Setting up Spark locally
- Setting Spark up on a cluster
- Introduction to Spark
- Working with RDDs (transforms and actions)
- Using `pyspark`

#### Objectives
- Intuition for how distributed computing works
- Working with Spark Resilient Distributed Datasets (RDD)
- Minimizing data shuffles: Accumulators, broadcast values and partitioning

### Lecture 22: Conserving RAM

- Efficient storage for numeric data (dense and sparse arrays)
- Efficient storage of stings (tries)
- Distinct value sketches and probabilistic data structures
- Yes/No oracle - the Bloom filter

#### Objectives
- Understand hardware latencies and why modern CPUs are starved
- Working with massive data sets without running out of memory

### Lecture 23: Distributed Computing

- Spark Streaming
- Spark SQL, DataFrames and DataSets
- NoSQL and big data formats and databases

#### Objectives
- Stateless and stateful processing of streaming data
- Using Spark SQL and its efficient data structures

### Lecture 24: Distributed Computing

- Using Spark MLLib with examples
- Using Spark GraphX with examples

#### Objectives
- Machine learning with `MLLib`
- Graph algorithms with `GraphX`

Laboratory Sequence (subejct to revision)
----
### Lab 1
- Reproducible analysis with
  - version control (`git`)
  - virtual environments (`conda`)
  - literate programming  (`Jupyter`)
  - testing (`doctesst`, `unittest`, `hypothesis`)
- Review of homework 1

### Lab 2
- Data science with Python
- Review of homework 2

### Lab 3
- Numerical recipes with Python
- Review of homework 3

### Lab 4
- Topic models
- Review of homework 4

### Lab 5
- Symbolic algebra with `sympy` and `theano`
- Review of homework 5

### Lab 6 (Mid-terms)
- Coding challenges

### Lab 7
- One arm bandits
- Review of homework 6

### Lab 8
- Optimizing distance matrix calculations
- Review of homework 7

### Lab 9
- Dynamic programming
- Review of homework 8

### Lab 10
- Introduction to C Part 1
- Review of homework 9

### Lab 11
- Introduction to C Part 2
- Review of homework 10

### Lab 12 (Finals)
- Final project presentations

### Possible alternative topics
- Gaussian processes
- Latent Dirichlet Allocation
- Hierarchical Dirichlet process
- Computer vision with `cv2`
- Machine learning with `scikit-learn`
- Packaging and distributing Python applications

Removed in revision
-----

### Lecture 19: Improving performance (GPU programming)

- CUDA 2
- `numba`

#### Objectives
- GPU hardware concepts
- Understand memory hierarchy
- Grids, blocks and threads
- CUDA kernels
- Using `numba` for easy CUDA
- First CUDA program

### Lecture 20: Improving performance (GPU programming)

- CUDA 2
- `numba`

#### Objectives
- Code matrix multiplication routines without shared memory
- Code matrix multiplication routines with shared memory

### Lecture 21: Improving performance (GPU programming)

- CUDA 3
- `numba`

#### Objectives
- Coding a Gaussian mixture model with CUDA kernels
