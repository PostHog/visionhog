from posthog import Posthog

posthog = Posthog('phc_fsN5YLls8XePWY7TXRo4RsZlLpErvzhDCcumYfmNU0K', host='http://localhost:8059')

posthog.capture("test", 'distinct_id')