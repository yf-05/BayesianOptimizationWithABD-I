# Accelerating Alloy Discovery on Small Datasets: A Bayesian Optimization Framework Guided by a Novel Efficiency Metric
Key codes and datasets of 'Accelerating Alloy Discovery on Small Datasets: A Bayesian Optimization Framework Guided by a Novel Efficiency Metric'

<div align="center">
  <img width="823" height="380" alt="image" src="https://github.com/user-attachments/assets/17246d6d-8814-457a-b74d-a25df2919d92" />
</div>


Conventional frameworks train the surrogate model *f* in isolation on the initial training pool and subsequently select an acquisition function independently. In contrast, the proposed framework simultaneously optimizes both the surrogate model and the acquisition function by directly minimizing the ABD-I metric on the current training data *X*, ensuring their synergy and maximizing overall discovery efficiency. The optimized pair is then used to predict the mean *μ* and uncertainty *σ* for all candidates in *X'* and select the next experiment point. 

The proposed metric ABD-I can be calculate via the function:

$$
\text{ABD-I} = \int_0^1 \left( f(x) - x \right) \, dx \quad \in [0, 0.5]
$$

In the paper, we demonstrate the acceleration of our new framework through two cases: Case 1 involves the Coefficient of Thermal Expansion (CTE) of High Entropy Alloys (HEAs), and Case 2 focuses on the CTE of Ni-based superalloys. We calculated the ABD-I for each case using both the conventional framework and our proposed framework, with the corresponding calculations available in the four Python files. The results strongly support that our proposed framework outperforms the conventional approach.
