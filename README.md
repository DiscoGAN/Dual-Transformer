# Dual-Transformer

This repository contains scripts for evaluating the trained Dual-Transformer model for water level prediction in Lake Superior, Lake Michigan-Huron, and Lake Erie.

## Instructions for Running the Code

**Reminder**: If you want to run the code for a specific lake, please execute the related script within the corresponding directory.  
For example, to evaluate the model performance for **Lake Superior**, run `main_transformer.py` in the `Lake_Superior_trained` directory.

### Lake Superior
- Run `main_Transformer.py` in the `Lake_Superior_trained` directory to test the pre-trained model.
- The prediction time scale can be adjusted by modifying the parameter `n` in the configuration class: 0 for 7 days, 1 for 32 days, 2 for 63 days, 3 for 91 days, 4 for 102 days, 5 for 147 days, and 6 for 180 days

### Lake Michigan-Huron
- Run `main_Transformer.py` in the `Lake_Michigan_Huron_trained` directory to test the pre-trained model.
- We only provide a trained model for a **180-day prediction time scale**, so set `n = 6` in the configuration class.

### Lake Erie
- Run `main_Transformer.py` in the `Lake_Erie_trained` directory to test the pre-trained model.
- Similar to Lake Michigan-Huron, we only provide a trained model for a **180-day prediction time scale**, so set `n = 6`.

---

For inquiries regarding **training details** or **further evaluation**, please contact 📧 pexue@mtu.edu.
