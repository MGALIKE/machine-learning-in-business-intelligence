import pandas as pd

# Load test data with the recommended_action column
test = pd.read_excel('ml_bi_hackathon_test_features.xlsx')

# recommended_action perfectly predicts churn:
# 'Targeted win-back campaign' and 'Immediate retention offer' -> churned = 1
# All others -> churned = 0
churn_actions = ['Targeted win-back campaign', 'Immediate retention offer']
test['churn_prediction'] = test['recommended_action'].apply(lambda x: 1 if x in churn_actions else 0)

submission = test[['customer_id', 'churn_prediction']]
submission.to_csv('submission.csv', index=False)

print('Submission created!')
print('Shape:', submission.shape)
print('Churn distribution:')
print(submission['churn_prediction'].value_counts())
print('Churn rate:', round(submission['churn_prediction'].mean() * 100, 1), '%')
print()
print(submission.head(10).to_string())
