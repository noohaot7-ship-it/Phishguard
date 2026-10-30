import tkinter as tk
from tkinter import ttk, messagebox
import re
import math
from collections import Counter, defaultdict

# AI-Based Phishing Email Awareness Detector
# Beginner-friendly mini project: Tkinter + Naive Bayes implemented in Python.
# No third-party packages are required.

TRAINING_DATA = [
    ("phishing", "urgent verify your account immediately click this link"),
    ("phishing", "your account will be suspended confirm your password now"),
    ("phishing", "you won a prize claim your reward by clicking here"),
    ("phishing", "security alert unusual login verify your identity"),
    ("phishing", "update your bank details to avoid account closure"),
    ("phishing", "payment failed open the link and enter card information"),
    ("phishing", "your package is waiting pay a small delivery fee"),
    ("phishing", "reset your password immediately using this link"),
    ("phishing", "congratulations you have won money provide your details"),
    ("phishing", "invoice attached enable macros and sign in"),
    ("phishing", "act now limited time offer verify billing information"),
    ("phishing", "dear customer your account is locked click here to unlock"),
    ("phishing", "confirm your password and one time code immediately"),
    ("phishing", "we detected suspicious activity login to secure account"),
    ("phishing", "refund pending submit bank account details"),
    ("legitimate", "hello team the meeting is scheduled for tomorrow at ten"),
    ("legitimate", "please find the class notes attached for your reference"),
    ("legitimate", "thank you for attending the project review meeting"),
    ("legitimate", "the assignment deadline is friday contact me if you need help"),
    ("legitimate", "your library book is due next week"),
    ("legitimate", "hi please review the attached report before our meeting"),
    ("legitimate", "the department seminar starts at nine in the auditorium"),
    ("legitimate", "your order has been shipped and can be tracked in your account"),
    ("legitimate", "monthly statement is available through the official banking app"),
    ("legitimate", "please reply with your preferred time for the discussion"),
    ("legitimate", "the course materials have been uploaded to the student portal"),
    ("legitimate", "your appointment is confirmed for monday afternoon"),
    ("legitimate", "here are the minutes from today's project meeting"),
    ("legitimate", "thanks for your feedback we will discuss it next week"),
    ("legitimate", "the event registration form closes on friday"),
]

SUSPICIOUS_WORDS = {
    "urgent": "Uses urgent language to pressure the reader.",
    "immediately": "Creates pressure to act immediately.",
    "suspended": "Threatens account suspension.",
    "password": "Requests or mentions sensitive login information.",
    "verify": "Asks the reader to verify an account or identity.",
    "click here": "Uses a generic call to click a link.",
    "won": "Claims a prize or unexpected reward.",
    "prize": "Mentions a prize or reward that may be bait.",
    "refund": "Mentions a refund that may be used as bait.",
    "one time code": "Mentions a one-time verification code.",
    "bank details": "Mentions sensitive banking information.",
    "credit card": "Mentions payment-card information.",
    "locked": "Uses an account-lock threat.",
    "act now": "Pressures the reader to act quickly.",
    "confirm": "Requests confirmation of potentially sensitive details.",
}

URL_RE = re.compile(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", re.IGNORECASE)
SHORTENERS = ("bit.ly", "tinyurl.com", "t.co", "is.gd", "cutt.ly", "rb.gy")
SUSPICIOUS_TLDS = (".xyz", ".top", ".click", ".work", ".zip")


def tokenize(text):
    return re.findall(r"[a-z0-9']+", text.lower())


class NaiveBayesClassifier:
    """Small Multinomial Naive Bayes text classifier implemented from scratch."""

    def __init__(self, examples):
        self.classes = sorted({label for label, _ in examples})
        self.class_docs = Counter()
        self.word_counts = {label: Counter() for label in self.classes}
        self.total_words = Counter()
        self.vocabulary = set()

        for label, text in examples:
            words = tokenize(text)
            self.class_docs[label] += 1
            self.word_counts[label].update(words)
            self.total_words[label] += len(words)
            self.vocabulary.update(words)

    def predict(self, text):
        words = tokenize(text)
        vocab_size = max(1, len(self.vocabulary))
        total_docs = sum(self.class_docs.values())
        scores = {}

        for label in self.classes:
            # Log probabilities avoid numerical underflow.
            score = math.log(self.class_docs[label] / total_docs)
            denominator = self.total_words[label] + vocab_size
            for word in words:
                count = self.word_counts[label][word]
                score += math.log((count + 1) / denominator)
            scores[label] = score

        highest = max(scores.values())
        exp_scores = {label: math.exp(value - highest) for label, value in scores.items()}
        total = sum(exp_scores.values()) or 1
        probabilities = {label: value / total for label, value in exp_scores.items()}
        prediction = max(scores, key=scores.get)
        return prediction, probabilities


def find_rule_warnings(text):
    lower = text.lower()
    warnings = []

    for phrase, explanation in SUSPICIOUS_WORDS.items():
        if phrase in lower:
            warnings.append(explanation)

    urls = URL_RE.findall(text)
    for url in urls:
        cleaned = url.lower().rstrip(".,;:!?)")
        if any(shortener in cleaned for shortener in SHORTENERS):
            warnings.append("Contains a shortened URL, which hides the final destination.")
        if any(tld in cleaned for tld in SUSPICIOUS_TLDS):
            warnings.append("Contains a domain ending sometimes used in suspicious links.")
        if "@" in cleaned:
            warnings.append("URL contains an @ symbol, which can be used to disguise a destination.")

    if re.search(r"\b(password|otp|one.time code|pin)\b", lower) and re.search(
        r"\b(send|share|reply|provide|enter|confirm|submit)\b", lower
    ):
        warnings.append("Appears to request confidential authentication information.")

    # Remove duplicate explanations while preserving order.
    return list(dict.fromkeys(warnings))


class PhishingDetectorApp:
    BG = "#F4F1FA"
    CARD = "#FFFFFF"
    INK = "#28233A"
    MUTED = "#716B82"
    PURPLE = "#7357C8"
    PURPLE_LIGHT = "#EEE8FF"
    RED = "#B9384B"
    RED_LIGHT = "#FFF0F1"
    GREEN = "#237A57"
    GREEN_LIGHT = "#EAF8F0"
    BORDER = "#E5DFF0"

    def __init__(self, root):
        self.root = root
        self.root.title("PhishGuard | Email Awareness Detector")
        self.root.geometry("1000x760")
        self.root.minsize(780, 650)
        self.root.configure(bg=self.BG)
        self.classifier = NaiveBayesClassifier(TRAINING_DATA)
        self.animation_job = None
        self.build_ui()

    def build_ui(self):
        header = tk.Frame(self.root, bg=self.BG)
        header.pack(fill="x", padx=28, pady=(22, 12))

        tk.Label(
            header, text="PhishGuard", font=("Segoe UI", 27, "bold"),
            bg=self.BG, fg=self.INK
        ).pack(anchor="w")
        tk.Label(
            header,
            text="AI-Based Phishing Email Awareness Detector",
            font=("Segoe UI", 11), bg=self.BG, fg=self.MUTED
        ).pack(anchor="w", pady=(2, 0))

        content = tk.Frame(self.root, bg=self.BG)
        content.pack(fill="both", expand=True, padx=28, pady=(0, 22))

        input_card = tk.Frame(content, bg=self.CARD, highlightbackground=self.BORDER,
                              highlightthickness=1)
        input_card.pack(fill="x", pady=(0, 14))

        tk.Label(input_card, text="EMAIL CONTENT", font=("Segoe UI", 10, "bold"),
                 bg=self.CARD, fg=self.INK).pack(anchor="w", padx=18, pady=(16, 6))

        self.email_box = tk.Text(
            input_card, height=7, wrap="word", font=("Segoe UI", 11),
            bg="#FCFBFE", fg=self.INK, insertbackground=self.INK,
            relief="flat", padx=12, pady=10, undo=True
        )
        self.email_box.pack(fill="x", padx=18)
        self.email_box.insert(
            "1.0",
            "Dear customer, your account will be suspended. Verify your password immediately "
            "by clicking this link: http://bit.ly/example"
        )

        button_row = tk.Frame(input_card, bg=self.CARD)
        button_row.pack(fill="x", padx=18, pady=14)

        self.analyze_button = tk.Button(
            button_row, text="Analyze email", command=self.analyze,
            font=("Segoe UI", 10, "bold"), bg=self.PURPLE, fg="white",
            activebackground="#6246B5", activeforeground="white",
            relief="flat", padx=18, pady=9, cursor="hand2"
        )
        self.analyze_button.pack(side="left")

        tk.Button(
            button_row, text="Try a safe example", command=self.load_safe_example,
            font=("Segoe UI", 10), bg=self.PURPLE_LIGHT, fg=self.INK,
            activebackground="#E1D8FF", relief="flat", padx=14, pady=9,
            cursor="hand2"
        ).pack(side="left", padx=(9, 0))

        tk.Button(
            button_row, text="Clear", command=self.clear,
            font=("Segoe UI", 10), bg="#F3F0F7", fg=self.INK,
            activebackground="#E8E2EF", relief="flat", padx=16, pady=9,
            cursor="hand2"
        ).pack(side="right")

        process_card = tk.Frame(content, bg=self.CARD, highlightbackground=self.BORDER,
                                highlightthickness=1)
        process_card.pack(fill="x", pady=(0, 14))
        tk.Label(process_card, text="ANALYSIS PIPELINE", font=("Segoe UI", 10, "bold"),
                 bg=self.CARD, fg=self.INK).pack(anchor="w", padx=18, pady=(13, 8))

        self.steps = []
        step_row = tk.Frame(process_card, bg=self.CARD)
        step_row.pack(fill="x", padx=18, pady=(0, 14))
        labels = ["1  Email input", "2  ML classification", "3  Rule checks", "4  Result"]
        for index, label in enumerate(labels):
            step = tk.Label(
                step_row, text=label, font=("Segoe UI", 9, "bold"),
                bg="#F2EFF7", fg=self.MUTED, padx=10, pady=10
            )
            step.pack(side="left", fill="x", expand=True, padx=(0 if index == 0 else 6, 0))
            self.steps.append(step)

        result_card = tk.Frame(content, bg=self.CARD, highlightbackground=self.BORDER,
                               highlightthickness=1)
        result_card.pack(fill="both", expand=True)

        result_header = tk.Frame(result_card, bg=self.CARD)
        result_header.pack(fill="x", padx=18, pady=(15, 4))
        tk.Label(result_header, text="ANALYSIS RESULT", font=("Segoe UI", 10, "bold"),
                 bg=self.CARD, fg=self.INK).pack(side="left")
        self.status_badge = tk.Label(
            result_header, text="Waiting for email", font=("Segoe UI", 9, "bold"),
            bg="#F2EFF7", fg=self.MUTED, padx=10, pady=5
        )
        self.status_badge.pack(side="right")

        self.result_title = tk.Label(
            result_card, text="Your results will appear here",
            font=("Segoe UI", 17, "bold"), bg=self.CARD, fg=self.INK, anchor="w"
        )
        self.result_title.pack(fill="x", padx=18, pady=(8, 3))

        self.confidence_label = tk.Label(
            result_card, text="ML confidence: —",
            font=("Segoe UI", 10), bg=self.CARD, fg=self.MUTED, anchor="w"
        )
        self.confidence_label.pack(fill="x", padx=18, pady=(0, 8))

        self.progress = ttk.Progressbar(result_card, mode="determinate", maximum=100)
        self.progress.pack(fill="x", padx=18, pady=(0, 12))

        tk.Label(result_card, text="Warning signs and rule-based reasoning",
                 font=("Segoe UI", 10, "bold"), bg=self.CARD, fg=self.INK).pack(
                     anchor="w", padx=18, pady=(0, 5)
                 )

        self.warnings_box = tk.Text(
            result_card, height=5, wrap="word", font=("Segoe UI", 10),
            bg="#FCFBFE", fg=self.INK, relief="flat", padx=10, pady=8,
            state="disabled"
        )
        self.warnings_box.pack(fill="both", expand=True, padx=18, pady=(0, 12))

        tk.Label(
            content,
            text="Educational prototype only. It can make mistakes; never rely on it as the only security check.",
            font=("Segoe UI", 9), bg=self.BG, fg=self.MUTED, wraplength=900, justify="left"
        ).pack(anchor="w", pady=(9, 0))

    def set_step(self, active):
        for i, label in enumerate(self.steps):
            if i < active:
                label.configure(bg=self.GREEN_LIGHT, fg=self.GREEN)
            elif i == active:
                label.configure(bg=self.PURPLE_LIGHT, fg=self.PURPLE)
            else:
                label.configure(bg="#F2EFF7", fg=self.MUTED)

    def analyze(self):
        text = self.email_box.get("1.0", "end").strip()
        if not text:
            messagebox.showinfo("Email needed", "Paste some email text first.")
            return

        if self.animation_job:
            self.root.after_cancel(self.animation_job)
            self.animation_job = None

        prediction, probabilities = self.classifier.predict(text)
        warnings = find_rule_warnings(text)
        # Combine model output and explicit warning rules for a cautious final result.
        flagged = prediction == "phishing" or len(warnings) >= 2
        confidence = probabilities[prediction] * 100

        self.analyze_button.configure(state="disabled", text="Analyzing...")
        self.status_badge.configure(text="Analyzing", bg=self.PURPLE_LIGHT, fg=self.PURPLE)
        self.result_title.configure(text="Checking email signals...", fg=self.INK)
        self.confidence_label.configure(text="ML confidence: calculating...")
        self.progress["value"] = 0
        self.show_warnings(["The email has been received."])
        self.set_step(0)

        self._pending_result = (flagged, prediction, confidence, warnings, probabilities)

        def advance(stage):
            if stage == 1:
                self.set_step(1)
                self.show_warnings(["Step 2: Naive Bayes compares words with example email patterns."])
            elif stage == 2:
                self.set_step(2)
                self.show_warnings(["Step 3: Checking suspicious phrases and URL patterns."])
            elif stage == 3:
                self.set_step(3)
                self.show_warnings(["Step 4: Combining the ML prediction with rule-based warnings."])
            else:
                self.set_step(4)
                self.display_result(*self._pending_result)
                self.analyze_button.configure(state="normal", text="Analyze email")
                self.animation_job = None
                return

            self.animation_job = self.root.after(550, lambda: advance(stage + 1))

        self.animation_job = self.root.after(250, lambda: advance(1))

    def display_result(self, flagged, prediction, confidence, warnings, probabilities):
        # This method is wrapped once by analyze() to display the pending result.
        self.progress["value"] = round(confidence)
        if flagged:
            self.status_badge.configure(text="CAUTION", bg=self.RED_LIGHT, fg=self.RED)
            self.result_title.configure(
                text="Potential phishing email detected", fg=self.RED
            )
        else:
            self.status_badge.configure(text="Lower risk", bg=self.GREEN_LIGHT, fg=self.GREEN)
            self.result_title.configure(
                text="No strong phishing signals detected", fg=self.GREEN
            )

        self.confidence_label.configure(
            text=f"ML prediction: {prediction.title()}  |  Model confidence: {confidence:.1f}%"
        )

        details = []
        if warnings:
            details.extend("• " + item for item in warnings)
        else:
            details.append("• No configured warning signs were detected by the rule checks.")

        if prediction == "phishing":
            details.append("• The Naive Bayes model found the text more similar to phishing examples.")
        else:
            details.append("• The Naive Bayes model found the text more similar to legitimate examples.")

        details.append("• Verify the sender and open the official website directly instead of trusting unexpected links.")
        self.show_warnings(details)

    def show_warnings(self, lines):
        self.warnings_box.configure(state="normal")
        self.warnings_box.delete("1.0", "end")
        self.warnings_box.insert("1.0", "\n".join(lines))
        self.warnings_box.configure(state="disabled")

    def load_safe_example(self):
        self.email_box.delete("1.0", "end")
        self.email_box.insert(
            "1.0",
            "Hello team, please review the attached project report before our meeting tomorrow. "
            "Thank you for your feedback."
        )

    def clear(self):
        if self.animation_job:
            self.root.after_cancel(self.animation_job)
            self.animation_job = None
        self.email_box.delete("1.0", "end")
        self.status_badge.configure(text="Waiting for email", bg="#F2EFF7", fg=self.MUTED)
        self.result_title.configure(text="Your results will appear here", fg=self.INK)
        self.confidence_label.configure(text="ML confidence: —")
        self.progress["value"] = 0
        self.show_warnings([])
        self.set_step(-1)
        self.analyze_button.configure(state="normal", text="Analyze email")


def main():
    root = tk.Tk()
    app = PhishingDetectorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
