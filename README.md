# EqualVoteBox 🧬

**EqualVoteBox** is a specialized Python library for medical ML transparency. It reveals the internal "split votes" within Random Forest models to prevent dangerous "majority rule" errors in clinical decision-making.

## 🛑 The Problem: Blind Majority Voting
In a standard **Random Forest**, the final prediction is based on majority voting. However, in medicine, a "majority" can be risky. 

**Example:**
Imagine a patient who requires **Drug B**, but the majority of trees vote for **Drug A**. 
* **Standard ML:** Predicts Drug A (High Risk).
* **The Clinical Gap:** If 20% of trees voted for Drug B because of a specific "red flag" feature, a doctor needs to see that dissent.

A key example is the **Na_to_K (Sodium-to-Potassium)** ratio. In many datasets, a ratio of **15.0** is a critical "cliff." If a patient falls just below this threshold, standard voting can become unstable.

## ✅ The Solution: EqualVoteBox
Launched on March 14, 2026, by **Aksheita Dholakia**, this library ensures every internal vote gets equal consideration and visibility.

- **Dissent Visibility:** See exactly how many trees disagreed with the final prediction.
- **Feature Transparency:** Helps doctors identify which specific features (like Na_to_K) triggered the internal votes.
- **Informed Decisions:** Moves from "Blind Trust" in a computer score to "Informed Verification" by a medical professional.

from evb_lib import EqualVoteBox

# Initialize the tool
evb = EqualVoteBox()

# Get the internal votes for a patient
# (Assuming your function is named 'get_votes')
votes = evb.get_votes(your_model, patient_data)

print(f"Medical Vote Split: {votes}")


## 🚀 Quick Start
```bash
pip install EqualVoteBox


