## 🚀 Instructions for Running the Scripts

This repository contains scripts for evaluating the trained Dual-Transformer model for seasonal water level prediction in Lake Superior, Lake Michigan-Huron, and Lake Erie. It also includes 2-layer and 4-layer LSTM models, Single-Transformer, and Complex-Transformer (Dual-Transformer without MLP) to generate six-month-ahead comparison results for Lake Superior.

---

## ⚠️ Reminder  
To run the code for a specific lake, please ensure the following:
1. Ensure that the scripts you intend to run are located in the appropriate directory.
2. Execute the scripts from within their respective directories.
3. For example, to evaluate the model performance for Lake Superior, select `main_transformer.py` from the `Lake_Superior_trained` directory and run it within `Lake_Superior_trained`.

---

### 🔹 **Lake Superior**
- Dual-Transformer
  - Run `main_Transformer.py` in the `Lake_Superior_trained` directory to test the pre-trained model.
  - The prediction time scale can be adjusted by modifying the parameter `n` in the configuration class: 0 for 7 days, 1 for 32 days, 2 for 63 days, 3 for 91 days, 4 for 102 days, 5 for 147 days, and 6 for 180 days
- LSTM (Only 6-month-ahead prediction)
  - Run `main_LSTM.py` in the `Lake_Superior_trained` directory to test the pre-trained model.
  - Adjust the **LSTM layer number** in the configuration class:
    - `num_layers = 2` for a **2-layer LSTM**
    - `num_layers = 4` for a **4-layer LSTM**
- Single-Transformer (Only 6-month-ahead prediction)
  - Run `main_single_Transformer.py` in the `Lake_Superior_trained` directory to test the pre-trained model.
- Complex-Transformer (Only 6-month-ahead prediction)
  - Run `main_complex_Transformer.py` in the `Lake_Superior_trained` directory to test the pre-trained model.

### 🔹 **Lake Michigan-Huron**
- Run `main_Transformer.py` in the `Lake_Michigan_Huron_trained` directory to test the pre-trained model.
- We only provide a trained model for a **180-day prediction time scale**, so set `n = 6` in the configuration class.

### 🔹 **Lake Erie**
- Run `main_Transformer.py` in the `Lake_Erie_trained` directory to test the pre-trained model.
- Similar to Lake Michigan-Huron, we only provide a trained model for a **180-day prediction time scale**, so set `n = 6`.

---

## 📬 Contact  
For inquiries regarding **training details** or **further evaluation**, please contact 📧 pexue@mtu.edu.
