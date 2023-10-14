from model_detection import model_detection

def main():
    model_detection(1, "./plank/LR_model.pkl", "./plank/input_scaler.pkl")

if __name__ == "__main__":
    main()