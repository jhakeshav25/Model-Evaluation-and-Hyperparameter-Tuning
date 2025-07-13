<p align="center">
  <img src="assets/banner.png" alt="Model Evaluation & Hyperparameter Tuning Banner"/>
</p>

# 🚀 Model Evaluation & Hyperparameter Tuning Toolkit

![CI](https://github.com/jhakeshav25/Model-Evaluation-and-Hyperparameter-Tuning/actions/workflows/main.yml/badge.svg)

A modular and production-ready toolkit for evaluating machine learning models using accuracy, precision, recall, F1-score, and cross-validation. Includes advanced hyperparameter tuning (GridSearchCV, RandomizedSearchCV) and ensemble strategies (Voting, Stacking).

---

## ✨ Features

✅ Evaluate models using:
- Accuracy, Precision, Recall, F1-score
- Cross-Validation

✅ Tune parameters with:
- GridSearchCV 🔍
- RandomizedSearchCV 🎲

✅ Ensemble methods:
- VotingClassifier, StackingClassifier

✅ Visual comparisons:
- Bar plots of model performance

---

## 📦 Installation

```bash
git clone https://github.com/jhakeshav25/Model-Evaluation-and-Hyperparameter-Tuning.git
cd Model-Evaluation-and-Hyperparameter-Tuning
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## ▶️ How to Run

```bash
python src/main.py
```

Or import functions into your own scripts:

```python
from src.evaluation import evaluate_model
from src.tuning import grid_search_tuning
from src.visualization import plot_metrics
```

---

## 📚 Notebooks

- `1. Model Evaluation.ipynb`
- `2. Hyperparameter Tuning.ipynb`
- `3. Advanced Techniques.ipynb`

---

## 📊 Metrics Comparison

| Metric    | Description                     | Best Use Case               |
|-----------|---------------------------------|-----------------------------|
| Accuracy  | Overall correctness             | When classes are balanced   |
| Precision | Positive prediction quality     | When false positives matter |
| Recall    | Coverage of true positives      | When false negatives matter |
| F1-score  | Harmonic mean of precision/recall | Imbalanced datasets       |

---

## 🗂 Project Structure

```
Model-Evaluation-and-Hyperparameter-Tuning/
├── assets/
│   └── banner.png
├── data/
├── notebooks/
├── results/
├── src/
│   └── main.py
├── .github/
│   └── workflows/
│       └── main.yml
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
```

---

## 👨‍💻 Author

**Keshav Kumar Jha**  
📧 [keshavkumarjha528@gmail.com](mailto:keshavkumarjha528@gmail.com)  
📍 Greater Noida, India  
🔗 [GitHub](https://github.com/jhakeshav25) • [LinkedIn](https://linkedin.com/in/keshav-kumar-jha-aa560022a/) • [LeetCode](https://leetcode.com/u/jhakeshav25/) • [GFG](https://www.geeksforgeeks.org/user/jhakeshav25/)

---

## 📜 License

MIT License © 2025 [Keshav Kumar Jha](https://github.com/jhakeshav25)

---

## 🤝 Contributing

Contributions, improvements, and feature suggestions are welcome!  
Please fork this repo and submit a pull request.
