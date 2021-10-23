import numpy as np

a = np.array([[1,2,3], [4,5,6], [7,8,9]])

unique_value = 0
for i in range(0, 3):
    for j in range(0, 3):
        unique_value += (10**(i*3+j))*(a[i][j])

print(unique_value)



























 # score = 0

    # for i in range(1):
    #     env.reset()
    #     for _ in range(10):
    #         env.render()
    #         # action: 0~8
    #         rand = random.randint(0,8)
    #         while rand in mylist:
    #             rand = random.randint(0,8)

    #         observation, reward, done, _ = env.step(rand)
    #         print(observation, reward)
    #         score += reward

    #         if done:
    #             break
    #         mylist.append(rand)
    # print(score)
