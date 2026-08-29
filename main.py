from model import train_logreg_model, train_svm_model, train_bert_model
from reply import generate_reply

print("Select model:")
print("1. Logistic Regression")
print("2. Support Vector Machine")
print("3. BERT") 
choice = input("Enter 1, 2, or 3: ")

if choice == "1":
    model, _ = train_logreg_model()
elif choice == "2": 
    model, _ = train_svm_model()
elif choice == '3':
    model, _ = train_bert_model()
else: 
    print("Invalid choice. Please choose '1', '2' or '3'.")

while True:
    user_review = input("\nEnter a review (or type 'exit' to quit): ")
    if user_review.lower() == "exit":
        break
    
    predicted = model.predict([user_review])[0]
    reply = generate_reply(predicted)

    print(f"Predicted Type: {predicted}")
    print(f"Auto-reply: {reply}")
