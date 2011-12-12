from riglib.experiment import consolerun
from tasks import Dots

if __name__ == "__main__":
    options = {
        "penalty_time": 5,
        "ignore_time": 4, 
        "rand_start": (1, 10), 
        "reward_time": 5, 
        "timeout_time": 3,
        "trial_probs": [0, None], 
    }
    exp = consolerun(Dots, ("autostart","button","ignore_correctness"), **options)
