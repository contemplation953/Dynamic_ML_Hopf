# Dynamic_ML_Hopf

Numerical simulation of the paper "A Novel Dimensionality Reduction Approach by Integrating Dynamics Theory and Machine Learning"

Runs on Julia 1.8

## 1.Data description

This code generates all the simulation data and images of the experimental results in the paper. All the temporary data are saved as jld in the corresponding files.

## 2.Catalog Description

The code contains three experimental sections:  
**vdp directory**:vdp equation simulation  
**num directory**:bogie model simulation  
**noise directory**:Simulation of bogie model under the influence of noise  

The **inputdata** file under each experiment contains all the data used for training, and all the **outputdata** files contain the training results  



DYNAMIC_ML_HOPF  
│  *.jl
│  README.md   
├─**noisy**  
│  │  *.jl     
│  ├─inputdata     
│  └─outputdata   
├─**num**  
│  │  *.jl  
│  ├─inputdata     
│  └─outputdata   
└─**vdp**  
    │  *.jl  
    ├─inputdata      
    └─outputdata  

## 3.Code Description

For each experiment, there are three parts: generating simulation data, performing training, and drawing graphs.

(1)**normal_data.jl** should be executed first to generate data of normal from.  
(2)execute **gen_data.jl** to generate training data.  
(3)execute **num_ex.jl** to performing training process.   
(4)execute **plot_figure.jl** to plot the figures of results.

### 3.1 vdp

**gen_data.jl**:Generate simulation data  
**normal_data.jl**:Calculate the results of the normal form under different parameters in advance to improve the training efficiency  
**num_ex.jl**:performing training process  
**plot_figure.jl**:draw pictures  

### 3.2 num

**gen_data.jl**:Generate simulation data and draw the bifurcation Chart
**normal_data.jl**:Calculate the results of the normal form under different parameters in advance to improve the training efficiency  
**num_ex.jl**:performing training process  
**plot_figure.jl**:draw pictures  
**error_figure.jl**:draw resault of Training error

### 3.3 noisy

**gen_data.jl**:Generate simulation data  
**normal_data.jl**:Calculate the results of the normal form under different parameters in advance to improve the training efficiency  
**num_ex.jl**:performing training process  
**plot_figure.jl**:draw pictures  
**noise_figure.jl**:draw the original image of the noisy data