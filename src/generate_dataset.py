import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Create synthetic data for age classification
num_samples = 5000
age_groups = ['<18', '18-24', '25-34', '35-49', '50+']
comments = [
    # Teen comments
    'omg this is so lit', 'bruh moment fr', 'no cap this slaps', 'vibing with my besties', 'this is fire', 
    'lowkey obsessed with this', 'that\'s sus', 'yeet', 'slay queen', 'big mood',
    # Young adult comments
    'just finished my finals', 'apartment hunting is stressful', 'dating apps are the worst', 'need coffee to function', 
    'adulting is hard', 'my student loans are killing me', 'just meal prepped for the week', 
    'working remote has pros and cons', 'side hustle paying off', 'networking event tonight',
    # Adult comments
    'my kids are driving me crazy', 'mortgage rates are insane', 'work-life balance is important', 
    'looking forward to the weekend', 'need to schedule a doctor appointment', 'thinking about changing careers', 
    'home renovation project', 'investing for retirement', 'parent-teacher conference today', 'business trip next week',
    # Middle-aged comments
    'my daughter is graduating college', 'planning for retirement', 'empty nest syndrome is real', 
    'blood pressure medication', 'reunion with old friends', 'proud of my children\'s accomplishments', 
    'considering downsizing our home', 'grandchildren visiting this weekend', 'managing aging parents and grown kids', 
    'health insurance costs are rising',
    # Senior comments
    'retirement is wonderful', 'grandchildren are such a blessing', 'doctor appointments every week', 
    'remember when phones had cords', 'back in my day', 'technology moves too fast', 
    'golden years indeed', 'senior discount is the best part', 'arthritis acting up today', 'bridge club meeting'
]

# Generate synthetic data
np.random.seed(42)
data = []
for _ in range(num_samples):
    age_group_idx = np.random.randint(0, len(age_groups))
    age_group = age_groups[age_group_idx]
    
    # Select comment style based on age group
    comment_start_idx = age_group_idx * 10
    comment_base = comments[comment_start_idx + np.random.randint(0, 10)]
    
    # Add some randomness to comments
    words = comment_base.split()
    if np.random.random() < 0.3:  # 30% chance to modify
        if len(words) > 3:
            # Remove or add random words
            if np.random.random() < 0.5:
                words.pop(np.random.randint(0, len(words)))
            else:
                fillers = ['really', 'very', 'super', 'kinda', 'honestly', 'basically', 'literally', 'actually', 'totally', 'absolutely']
                words.insert(np.random.randint(0, len(words)), fillers[np.random.randint(0, len(fillers))])
    
    comment = ' '.join(words)
    
    # Add punctuation and capitalization variations
    if np.random.random() < 0.2:  # 20% chance for all lowercase
        comment = comment.lower()
    if np.random.random() < 0.1:  # 10% chance for all caps
        comment = comment.upper()
    if np.random.random() < 0.3:  # 30% chance to add extra punctuation
        comment += '!' * np.random.randint(1, 4)
    
    data.append({'comment': comment, 'age_group': age_group})

# Create DataFrame
df = pd.DataFrame(data)

# Split into train, validation, and test sets
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Save to CSV files
train_df.to_csv('/home/ubuntu/nlp_age_detection/data/raw/train_data.csv', index=False)
val_df.to_csv('/home/ubuntu/nlp_age_detection/data/raw/val_data.csv', index=False)
test_df.to_csv('/home/ubuntu/nlp_age_detection/data/raw/test_data.csv', index=False)
df.to_csv('/home/ubuntu/nlp_age_detection/data/raw/full_dataset.csv', index=False)

print(f'Created synthetic dataset with {len(df)} samples')
print(f'Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}')
print('\nSample data:')
print(df.head())
print('\nAge group distribution:')
print(df['age_group'].value_counts())
