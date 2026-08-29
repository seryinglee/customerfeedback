def generate_reply(label: str) -> str:
    replies = {
        "positive": "Thank you for your kind feedback! We're glad you had a good experience. 🌟",
        "negative": "We're sorry to hear about your experience. We'll work on improving. 🙏",
        "neutral": "Thank you for your feedback! We appreciate your input. 🙂",
        "critical": "We value your constructive criticism and will make improvements. ⚡"
    }
    return replies.get(label.lower(), replies["neutral"])
