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
git clone https://github.com/your-username/HRfH-Hackathon-2026.git
cd HRfH-Hackathon-2026

# Install dependencies
pip install -r requirements.txt

# Launch dashboard
streamlit run app.py
```
#### 2.1 Dashboard understanding 
Dashboard was created with the intention to be able to play with the variables, 
 
### 3. Start Analyzing

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

## Citation
@dataset{hrfh_hackathon_2026,
  title={Synthetic Knee Replacement Recovery Data},
  author={Health Research from Home},
  year={2026},
  url={https://github.com/your-username/HRfH-Hackathon-2026}
}

## Acknowledgments
Health Research from Home (HRfH)
Clinical advisors for medical validation
Hackathon participants

## License
MIT License - see [LICENSE](LICENSE) file




