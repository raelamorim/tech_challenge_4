from collections import defaultdict, deque

pose_history = defaultdict(lambda: deque(maxlen=30))
emotion_history = defaultdict(lambda: deque(maxlen=10))
