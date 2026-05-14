# Health-Research-from-Home-Hackathon-fantastic-4-
 HRfH Hackathon 2026: Synthetic Knee Replacement Recovery Data
 https://github.com/Health-Research-From-Home/HRfH-Hackathon-2026

# TASK1
## Project Overview
### Core Objective
Create a synthetic knee replacement (TKA) recovery dataset that realistically simulates wearable device data with missing values, noise, and patient heterogeneity for the HRfH Hackathon 2026 Data Analysis Challenge.

## Project Background
### Medical Context
1. Over 2 million knee replacement surgeries performed globally per year
2. Recovery trajectories are highly heterogeneous: patients recover over 3 months to 2+ years
3. Existing studies struggle to capture real-world recovery diversity

### Hackathon Challenge
Healthcare research institutions need a realistic synthetic dataset to:
1. Develop missing data imputation algorithms
2. demonstarte Patient clustering and phenotype discovery
3. Predict long-term recovery trajectories

## Technical Architecture
 ## Quick Stats

| Metric | Value |
|--------|-------|
| Patients | 1,000 |
| Days tracked | 365 |
| Daily records | 365,000 |
| Missing steps | ~34% |
| Recovery phenotypes | 3 (fast/intermittent/slow) |
| Missing mechanisms | 4 (MCAR, MAR, MNAR, Block) |

 ## Data description variables 
| Stat | patient_id | age	| height_cm | weight_kg | BMI | baseline_pain | battery_life_hours | last_active_days | max_steps |
|--------|-------|--------|-------|--------|-------|--------|-------|--------|-------|
| count	| 1000.000000 | 1000.000000	| 1000.000000 | 1000.000000 | 1000.000000 | 1000.000000 | 1000.000000 | 1000.000000 | 1000.000000 |
| mean |	499.500000	| 67.633000	| 168.174800	| 86.995500	| 30.628700	| 5.440000 | 24.082000 | 1.996000 | 5574.567000 |
| std	| 288.819436	| 7.881144	| 10.042092	| 18.936273	| 5.408541	| 1.992579	| 7.679076	| 2.162096	| 1471.501063 |
| min	| 0.000000	| 45.000000	| 145.000000	| 42.400000	| 18.000000	| 1.000000	| 8.000000	| 0.000000	| 2444.000000 |
| 25%	| 249.750000	| 62.000000	| 161.400000	| 72.975000	| 26.700000	| 4.000000	| 18.675000	| 0.000000	| 4473.750000 |
| 50%	| 499.500000	| 68.000000	| 168.200000	| 85.300000	| 30.000000	| 5.000000	| 23.850000	| 1.000000	| 5475.500000 |
| 75%	| 749.250000	| 73.000000	| 175.225000	| 99.100000	| 34.725000	| 7.000000	| 29.025000	| 3.000000	| 6594.250000 |
| max	| 999.000000	| 90.000000	| 200.000000	| 154.300000	| 45.900000	| 10.000000	| 49.200000	| 10.000000	| 10062.000000 |

##  Process Steps
### 1. Generate Data (in Colab)

Run this notebook:
[Hackathon Colab Notebook](https://colab.research.google.com/drive/1BVV8Q0xRlLwziRv-g54IM2hunn4F2PQZ)
[Hackthon Colab Notebook- missing data](https://colab.research.google.com/drive/1JA1zPx344pJpAkhD363n62Fo3GdQERbM)

Output: `synthetic_patients.csv` + `synthetic_daily_data.csv`

### 2. Install & Explore (Locally)
```bash
# Clone repo
git clone https://github.com/foluwa/HRfH-Hackathon-2026.git
cd HRfH-Hackathon-2026

# Install dependencies
pip install -r requirements.txt

# Launch dashboard
streamlit run app.py
```
## 3. Dashboard User's Guide

User can access the Dashboard scanning the following QR Code:

<a href="https://huggingface.co/spaces/Foluwa/HRfH-2026-Team-4-task-1" target="_blank" rel="noopener noreferrer">Dashboard Link</a>
![QR CODE](https://raw.githubusercontent.com/foluwa/HRfH-2026-Team-4/main/screenshots/qr_code.png)


Dashboard helps users interacting and generating a comma-separated values file (.csv) of a virtual subject yearly-step count. 

User can modify following parameters
- Age range (between 45 and 90)
- Gender (male, female, non-binary)
- Recovery phenotype (fast intermediatre and slow)
- Target of missingness (Missing complete at random, Missing at random, Missing not at random)

- 
 
## Dashboard Screenshots

The Streamlit dashboard provides an interactive interface for exploring simulated patient recovery trajectories, configurable missingness mechanisms, clustering, Monte Carlo evaluation, and downloadable outputs.


### Dashboard Overview and Controls

![Dashboard screenshot 3](https://raw.githubusercontent.com/foluwa/HRfH-2026-Team-4/main/screenshots/dashboard_3.png)

### Dashboard Visualisations and Analysis

![Dashboard screenshot 4](https://raw.githubusercontent.com/foluwa/HRfH-2026-Team-4/main/screenshots/dashboard_4.png)


### 4. Start Analyzing

## Key Features
Three Recovery Phenotypes
| Type | % Patients | Recovery Time | Characteristics |
|------|-----------|---------------|-----------------|
| Fast | 20% | 3-4 months | Young, active, low pain |
| Intermittent | 60% | 5-6 months | Typical ups & downs |
| Slow | 20% | 8-12 months | Older, overweight, high pain |

## Data Output
Two CSV files are generated in the repository root:
- **synthetic_patients.csv** (1,000 rows): Patient demographic and baseline characteristics
- **synthetic_daily_data.csv** (365,000 rows): Daily time-series measurements for each patient
- also an app?!

**Patient Age Distribution**
<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/1e8082eb-0407-4b52-978b-0d57586bb8e4" />

**Patient Phenotype Characteristics**
<img width="1589" height="490" alt="image" src="https://github.com/user-attachments/assets/6c5baeca-ca28-4fe9-b23d-1659377eb651" />
3 recovery phenotypes differ significantly across age, BMI, and functional capacity:
 Fast recoverers: younger, lower BMI, higher maximum steps (~7,500/day)
 Intermittent recoverers: middle-aged, moderate BMI, moderate steps (~7,000/day)
 Slow recoverers: older, higher BMI, lower maximum steps (~4,000/day)

 **Comorbidity Prevalence**
 <img width="1489" height="804" alt="image" src="https://github.com/user-attachments/assets/a4b273ca-7ec0-4d71-8686-73630f64c0d8" />
comorbidity patterns for knee replacement candidates:
 - Hypertension (79.9%) and cardiovascular disease (74.6%) are most prevalent
 - Diabetes (42.3%) is significant, known to impair post-operative recovery
 - Osteoporosis (7.8%) and musculoskeletal disease (22.9%) are less common
Comorbidities influence recovery trajectories in the simulation through slower logistic curves and higher missing data rates.

**QQ-plots of continuous baseline characteristics**
<img width="2506" height="1446" alt="qqplots" src="https://github.com/user-attachments/assets/24ef32c0-3e60-40f1-8319-88515dc5056c" />

Quantile–quantile (QQ) plots of the generated sample data for age, body mass index (BMI), max daily steps count and higher reported baseline pain against a theoretical normal distribution.

# Parameter Selection and Justification
All synthetic data parameters were selected to reflect published epidemiology, clinical practice patterns, and peer-reviewed outcome literature for total knee arthroplasty (TKA). This section documents the evidence basis for each parameter choice.

## Demographics
### Age (Truncated Normal: μ=68, σ=8, range 45–90 years)
**Justification:**
- **Population mean**: Published epidemiology shows average TKA age is 65–66 years in the United States
- **Standard deviation (8 years)**: Captures realistic spread where ~68% of patients fall between 60–76 years and ~95% fall between 52–84 years
- **Minimum (45 years)**: Patients under 50 are considered "young for knee replacement" but still occur; 45 years is a realistic lower bound for advanced osteoarthritis
- **Maximum (90 years)**: Some patients aged 85–90 still pursue TKA if medically fit; age alone is not a contraindication
- **Clinical impact**: Age directly increases comorbidity burden (via sigmoid functions) and slows recovery phenotype assignment, reflecting real-world outcomes where older patients recover more slowly

### Gender (Categorical: 45% male, 50% female, 5% non-binary)
**Justification:**
- **Female predominance (50%)**: TKA cohorts typically show slight female majority due to higher osteoarthritis prevalence and surgical acceptance in women
- **Male proportion (45%)**: Reflects actual gender distribution in surgical populations
- **Non-binary (5%)**: Included for demographic inclusivity and reflects modern healthcare diversity
- **Clinical impact**: Gender does not directly affect recovery in this model but ensures demographic realism

### Height (Truncated Normal: μ=168 cm, σ=10, range 145–200 cm)
**Justification:**
- **Mean (168 cm)**: Approximates average adult height in Western populations (UK/US data)
- **Standard deviation (10 cm)**: Captures realistic variation (~±10 cm covers most adults)
- **Range (145–200 cm)**: Accommodates short and tall individuals whilst excluding implausible extremes
- **Clinical impact**: Height is used to calculate BMI; affects functional capacity estimates

## Body Composition
### BMI (Mixture Distribution: 15% normal, 40% overweight, 45% obese)
**Justification:**
- **Obesity prevalence (45%)**: Reflects 40–50% obesity rates in published TKA cohorts (higher than general population)
- **Overweight (40%)**: Common presentation; BMI 25–30
- **Normal BMI (15%)**: Healthier subset who pursue preventive/early surgery
- **Three-category model**: More clinically realistic than single normal distribution; captures real-world bimodal distribution
- **Distribution within categories:**
  - Normal: μ=23, σ=2 (range 18–25)
  - Overweight: μ=28, σ=2 (range 25–30)
  - Obese: μ=35, σ=4 (range 30–50)

**Clinical impact**: BMI directly drives:
1. Comorbidity risk (sigmoid functions for diabetes, hypertension, cardiovascular disease)
2. Max steps calculation (higher BMI = lower functional ceiling)
3. Recovery phenotype assignment (higher BMI increases slow phenotype probability)

### Weight (Derived from BMI)
**Formula**: `weight_kg = BMI × (height_m²)`
**Justification**: Calculated from BMI and height using standard physiological formula, ensuring internal consistency

## Pre-operative Clinical Status
### Pre-operative Activity Level (Categorical: 35% low, 50% moderate, 15% high)
**Justification:**
- **Distribution**: Reflects typical TKA population where many patients have been activity-limited by arthritis pain
- **Low activity (35%)**: Patients with severe functional limitation before surgery
- **Moderate activity (50%)**: Most common; partial functional preservation pre-operatively
- **High activity (15%)**: Active older adults or younger TKA candidates
- **Clinical impact**: Higher pre-operative activity increases maximum steps and improves recovery phenotype probability (via sigmoid function with weight +0.5 × activity_code)

### Baseline Pain (Truncated Normal: μ=6, σ=2, range 1–10)
**Justification:**
- **Mean (6/10)**: Reflects moderate-to-severe pain that typically drives surgical decision-making
- **Range (1–10)**: Standard pain scale (0 = no pain, 10 = worst pain)
- **Clinical impact**: Pain directly correlates with post-operative sleep duration (sleep_hours reduced by 0.15 × baseline_pain), modelling how chronic pain disrupts recovery

## Comorbidities
All comorbidities use **sigmoid logistic functions** to model realistic age and BMI dependencies. This approach captures two key clinical patterns:
1. Higher age and BMI increase disease prevalence non-linearly
2. Comorbidities cluster (e.g., obese diabetic patients are more likely hypertensive)

### Cardiovascular Disease (Binary: 74.6% prevalence)
**Formula**: `p_cardiovascular = sigmoid(−7 + 0.05 × age + 0.08 × BMI)`
**Justification:**
- **Literature prevalence**: 20–40% in general TKA cohorts; our 74.6% reflects selection for older, heavier patients (mean age 68)
- **Age coefficient (0.05)**: Modest increase with age; cardiovascular disease is extremely common in 65+ populations
- **BMI coefficient (0.08)**: Obesity is a strong cardiovascular risk factor
- **Intercept (−7)**: Calibrated to match observed prevalence in synthetic cohort
- **Clinical impact**: Cardiovascular disease slows recovery rate (k parameter reduced by 0.003)

### Diabetes (Binary: 42.3% prevalence)
**Formula**: `p_diabetes = sigmoid(−6 + 0.18 × BMI)`
**Justification:**
- **BMI dependence**: Obesity is the strongest modifiable risk factor for type 2 diabetes
- **Coefficient (0.18)**: High sensitivity; reflects strong obesity-diabetes correlation
- **Literature prevalence**: 30–50% in TKA cohorts, especially in obese populations
- **Age independence**: Type 2 diabetes is already embedded via BMI; age correlation is indirect
- **Clinical impact**: Diabetes increases slow phenotype probability (via sigmoid: +0.8 × diabetes), reflecting impaired wound healing and slower functional recovery

### Hypertension (Binary: 79.9% prevalence)
**Formula**: `p_hypertension = sigmoid(−5 + 0.15 × BMI + 0.03 × age)`
**Justification:**
- **BMI and age dependent**: Both are independent risk factors
- **Literature prevalence**: 60–80% in TKA cohorts (we observe 79.9%)
- **Age coefficient (0.03)**: Modest effect; hypertension strongly correlates with age in 65+ populations
- **Clinical impact**: Hypertension slows recovery rate (k reduced by 0.002)

### Osteoporosis (Binary: 7.8% prevalence)
**Formula**: `p_osteoporosis = sigmoid(−5 + 0.04 × age)`
**Justification:**
- **Age only**: Primary risk factor is advancing age
- **Low prevalence (7.8%)**: Osteoporosis is less common than other comorbidities in mixed-gender TKA cohorts
- **Female predominance expected**: Women at higher risk, but synthetic cohort is ~50/50 gender split
- **Clinical impact**: Does not directly affect recovery in this model but adds realistic comorbidity burden

### Musculoskeletal Disease (Binary: 22.9% prevalence)

**Formula**: `p_msk = sigmoid(−2 + 0.03 × BMI)`

**Justification:**
- **BMI dependent**: Higher weight increases musculoskeletal burden
- **Literature**: Pre-existing musculoskeletal conditions (e.g., rheumatoid arthritis, other joint OA) present in ~15–25% of TKA candidates
- **Clinical impact**: Does not directly affect recovery parameters but reflects real-world comorbidity complexity

## Wearable Device Parameters
### Device Brand (Categorical: Fitbit, Apple Watch, Garmin, equal probability)
**Justification:**
- **Equal distribution**: Reflects market competition; no single dominant brand in real-world populations
- **Brands chosen**: Most common consumer wearables with step-counting capability
- **Clinical impact**: Device brand does not affect recovery simulation but adds realism for data interpretation

### Battery Life (Truncated Normal: μ=24 hours, σ=8, range 8–72 hours)
**Justification:**
- **Realistic range**: Modern smartwatches vary from 8-hour (daily charge) to 72-hour (multi-day) battery
- **Mean (24 hours)**: Many devices require daily charging
- **Clinical impact**: Battery life influences missing data probability (indirectly via last_active_days)

### Last Active Days (Truncated Normal: μ=2, σ=3, range 0–30 days)
**Justification:**
- **Mean (2 days)**: Most wearables are actively used; few patients abandon devices
- **Range (0–30)**: Some patients stop wearing devices after weeks (non-compliance)
- **Clinical impact**: Directly affects missing data probability via sigmoid function:
  - `p_missing = sigmoid(−4 + 0.03 × last_active_days)`
  - Recent device use (low last_active_days) = less missing data
  - Inactive devices (high last_active_days) = more missing data

## Recovery Phenotypes

### Phenotype Assignment (Categorical: 20% fast, 60% intermittent, 20% slow)
**Justification:**
Recovery phenotypes are assigned via **probability weighting** based on risk factors:

```python
fast_score = sigmoid(2 − 0.03 × age − 0.05 × BMI + 0.5 × activity_code)
slow_score = sigmoid(−4 + 0.04 × age + 0.08 × BMI + 0.8 × diabetes)
intermittent_score = 1.0 (baseline)
```

## Design Rationale

### Recovery Phenotypes (Fast, Intermittent, Slow)

The recovery phenotypes were designed to represent the heterogeneity observed in real post-operative rehabilitation trajectories.

- **Fast phenotype**: Represents patients with rapid and stable rehabilitation progress
- **Intermittent phenotype**: Captures fluctuating recovery with alternating periods of improvement and setbacks
- **Slow phenotype**: Represents patients with prolonged functional limitations and delayed rehabilitation

These phenotypes create meaningful structure in the dataset and allow clustering and trajectory analysis methods to identify distinct recovery patterns.

### Recovery Rate Parameter (k)

The recovery rate parameter controls how quickly a patient progresses through rehabilitation over time. Higher values of k generate steeper recovery curves that reach functional independence earlier, whilst lower values create slower and more prolonged recovery trajectories.

This parameter was intentionally varied across phenotypes to reflect clinically plausible rehabilitation speeds, where fast recovery patients improve more rapidly and slow recovery patients require substantially longer periods to achieve functional gains. Using different recovery rates also increases longitudinal variability within the dataset and prevents all trajectories from following identical temporal patterns.

### Recovery Inflection Point (t₀)

The inflection point parameter determines the time at which recovery accelerates most rapidly. This was included because rehabilitation does not progress linearly after surgery. Patients often experience an early low-function period immediately following the operation, followed by accelerated improvement during active rehabilitation before eventually plateauing.

Earlier inflection points were assigned to fast recovery patients to represent rapid rehabilitation engagement, whilst delayed inflection points were assigned to slow recovery patients to simulate prolonged recovery initiation and delayed functional improvement.

### Maximum Achievable Steps (max_steps)

The maximum achievable step count represents the patient's long-term functional mobility ceiling. This variable was designed to depend on:

- Age
- BMI
- Pre-operative activity level

These characteristics strongly influence rehabilitation outcomes and long-term physical function in real populations. Older age and higher BMI reduce expected mobility capacity, whilst higher baseline activity increases recovery potential. Phenotype-specific multipliers were also applied so that fast recovery patients achieve higher long-term mobility and slow recovery patients stabilise at lower functional levels.

### Latent Recovery States (Stable, Improving, Flare, Plateau)

The latent recovery states were introduced to model hidden physiological or behavioural recovery conditions that are not directly observable but influence wearable measurements.

Recovery following surgery is rarely smooth or perfectly continuous, as patients may experience temporary setbacks, pain flares, fatigue, or rehabilitation plateaus:

- **Stable state**: Consistent recovery
- **Improving state**: Periods of accelerated rehabilitation progress
- **Flare state**: Simulates temporary deterioration or pain episodes
- **Plateau state**: Represents stalled recovery

Including these states creates realistic temporal variability and prevents the trajectories from appearing overly deterministic.

### Transition Matrices

The transition matrices define the probability of patients moving between latent recovery states over time and were designed to reflect phenotype-specific rehabilitation dynamics:

- **Fast recovery patients**: Higher probabilities of remaining in stable or improving states; lower probabilities of prolonged flare episodes, reflecting more resilient rehabilitation behaviour
- **Slow recovery patients**: Greater persistence in flare and plateau states to simulate chronic instability and delayed recovery
- **Intermittent patients**: More balanced transition probabilities to create fluctuating trajectories with alternating periods of progress and deterioration

These matrices introduce temporal dependency and clinically plausible stochastic behaviour into the simulation.

### State Effect Multipliers

The state effect multipliers determine how latent recovery states influence observable daily activity levels:

- **Improving states**: Increase step counts to simulate successful rehabilitation progress and increased mobility
- **Flare states**: Substantially reduce activity to represent pain, inflammation, swelling, or reduced function
- **Plateau states**: Slightly reduce activity to reflect suboptimal but stable recovery

These multipliers were included to establish a realistic relationship between hidden recovery conditions and measurable wearable outcomes.

### Autocorrelated Noise

Autocorrelated noise was incorporated to simulate the temporal consistency commonly observed in wearable sensor data. In real rehabilitation trajectories, unusually good or poor recovery days are often followed by similar days rather than completely independent fluctuations.

The autocorrelation structure creates sustained periods of improvement or decline, smoother longitudinal behaviour, and more realistic recovery dynamics compared with purely random independent noise.

## Patient Characteristic Justifications

### Age

Age was included because it is a major determinant of rehabilitation outcomes following knee replacement surgery. Older patients generally experience slower recovery, lower physical resilience, reduced mobility, and a higher burden of comorbidities. In the simulator, age influences both maximum mobility potential and the probability of certain health conditions, helping generate clinically plausible recovery patterns across the synthetic population.

### BMI

BMI was incorporated because body composition strongly influences post-operative mobility, rehabilitation difficulty, cardiovascular load, and long-term physical function. Higher BMI values reduce maximum achievable step counts and increase the probability of metabolic and cardiovascular comorbidities within the simulation.

The BMI distribution was intentionally generated using mixture groups to better reflect realistic clinical populations rather than assuming a single normal distribution.

### Pre-operative Activity Level

Pre-operative activity level was included because baseline physical fitness and mobility are strong predictors of rehabilitation success after surgery. Patients with higher pre-operative activity are generally better conditioned, more physically resilient, and more likely to engage effectively with rehabilitation programmes. This variable therefore increases maximum achievable mobility and contributes to faster functional recovery trajectories.

### Baseline Pain

Baseline pain was designed to influence recovery-related behaviour such as sleep quality and overall rehabilitation burden. Higher baseline pain levels reduce simulated sleep duration and contribute indirectly to poorer recovery experiences. This variable was included to introduce additional behavioural realism into the synthetic trajectories and create more complex interactions between physiological and wearable-derived variables.

### Comorbidities

The comorbidities (diabetes, hypertension, cardiovascular disease, osteoporosis, musculoskeletal disease) were generated probabilistically based on age and BMI because these health conditions are not independent in real clinical populations. 

Conditions such as diabetes and cardiovascular disease increase rehabilitation complexity and are associated with reduced physical function and poorer long-term recovery outcomes. Including correlated comorbidities improves population realism and creates clinically plausible variation in recovery trajectories.

## Wearable Data Features

### Heart Rate

Heart rate was simulated as a function of baseline cardiovascular characteristics and physical activity levels. Higher step counts slightly increase heart rate, whilst additional random variation reflects physiological variability and wearable measurement noise. This relationship was included to mimic real-world wearable sensor behaviour and provide a secondary physiological signal alongside mobility trajectories.

### Sleep Duration

Sleep duration was designed to reflect the interaction between pain, recovery, and physical activity. Higher pain levels reduce sleep quality and duration, whilst greater physical activity slightly improves sleep outcomes. Random variability was added to reflect natural behavioural fluctuations. Including sleep data expands the simulator beyond mobility alone and creates a richer multimodal wearable dataset.

## Missing Data Mechanisms

The missingness pathways were intentionally designed to replicate realistic wearable data challenges commonly encountered in digital health research:

- **Random missingness**: Simulates occasional device or transmission failures
- **Block missingness**: Represents temporary device disengagement or charging gaps
- **Flare-related missingness**: Reflects patients being less likely to wear devices during periods of pain or deterioration
- **Seasonal missingness**: Simulates behavioural variation during winter months

Including multiple missingness mechanisms creates more realistic incomplete longitudinal datasets and allows evaluation of analytical methods under different missing data assumptions.

## Citation
@dataset{hrfh_hackathon_2026,
  title={Synthetic Knee Replacement Recovery Data},
  author={Health Research from Home},
  year={2026},
  url={https://github.com/foluwa/HRfH-Hackathon-2026}
}

## Acknowledgments
Health Research from Home (HRfH)
Clinical advisors for medical validation
Hackathon participants

## License
MIT License - see [LICENSE](LICENSE) file




