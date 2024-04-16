# 24510115_assign1

데이터 처리 과정(dataset.py) : 경로를 받아 올 때 train이 True면 하위경로를 train, 아니면 하위경로에 test를 붙여서 파일 구분  
                  모든 파일은 .png이므로 png로 끝나는 애들의 이름을 전부 리스트에 담은 뒤에 Image 함수를 통해서 gray-scaleing을 해주고 
                  무슨 숫자에 대한 이미지인가를 파일 이름을 parsing해서 label 이라는 변수에 담아서 차후 분석을 진행함
                  transforms을 통해서 평균0.1307 표준편차 0.3081 를 이용해 normalize를 진행
                  main에는 제대로 된건지 확인 위해서 shape를 가져오고, label 값을 확인하는 메서드 작성해놓음



모델 (model.py):Lenet5 모델에  regurlaztion 은 weight decay와(weight decay는 main에서 적용) dropout 이용 , 
                data agumentation은 이용하지 않음, 적용하지 않은 이유는 적용하지 않아도 성능이 준수해서
                적용하지 않음. Lenet5 input이 32x32 라고 해서 28x28 이미지 사이즈를 32x32로 바꿔주기 위해  초기 convlayer에 padding 2 해줌, 활성화 함수는 RELU 사용 
                CustomMLP는 기본적으로 위에서 만든 Lenet5 모델을 그대로 이용함, 파라메터 수를 아예 같게 해주려다 보니 다른 모델의 형태를 바꿔주는대 제약이 많고, 이미 Lenet5 성능 
                을 테스트 해보니 성능이 준수해서 뒤의 Fully conected layer만 변경 
                
                파라메터수 계산 :Le-net
                Convolutional Layer 1 (self.conv1):

                  Input 채널 수: 1
                  Output 채널 수: 6
                  커널 크기: 5x5
                  Bias 항목: 각 output 채널마다 하나
                  파라미터 수 = (입력 채널 수 * 커널 높이 * 커널 너비 + 1) * 출력 채널 수 = (1 * 5 * 5 + 1) * 6 = 156
                Convolutional Layer 2 (self.conv2):
                  
                  Input 채널 수: 6
                  Output 채널 수: 16
                  커널 크기: 5x5
                  Bias 항목: 각 output 채널마다 하나
                  파라미터 수 = (입력 채널 수 * 커널 높이 * 커널 너비 + 1) * 출력 채널 수 = (6 * 5 * 5 + 1) * 16 = 2416
                  
                Fully Connected Layer 1 (self.fc1):                  
                  입력 크기: 1655
                  출력 크기: 120
                  Bias 항목: 각 출력 노드마다 하나
                  파라미터 수 = (입력 크기 + 1) * 출력 크기 = (1655 + 1) * 120 = 48120
                  
                Fully Connected Layer 2 (self.fc2):
                  입력 크기: 120
                  출력 크기: 84
                  Bias 항목: 각 출력 노드마다 하나
                  파라미터 수 = (입력 크기 + 1) * 출력 크기 = (120 + 1) * 84 = 10164
                
      
                Fully Connected Layer 3 (self.fc3):
                  입력 크기: 84
                  출력 크기: 10 (최종 클래스 수)
                  Bias 항목: 각 출력 노드마다 하나
                  파라미터 수 = (입력 크기 + 1) * 출력 크기 = (84 + 1) * 10 = 850

                총합:  156 + 2,416 + 48,120 + 10,164 + 850 = 61,706개,코드로 확인해본 결과도 동일함


              Custom MLP 에서는 layer에 변형을 주면서  paramter 숫자를 Lenet 5 와 동일하게 유지하는게 어렵고, Lenet 성능이 잘나와서 위의 Lenet에서 dropout 비율만 바꿧다..
             

             파라메터수 계산 : Custom MLP
                Convolutional Layer 1 (self.conv1):

                  Input 채널 수: 1
                  Output 채널 수: 6
                  커널 크기: 5x5
                  Bias 항목: 각 output 채널마다 하나
                  파라미터 수 = (입력 채널 수 * 커널 높이 * 커널 너비 + 1) * 출력 채널 수 = (1 * 5 * 5 + 1) * 6 = 156
                Convolutional Layer 2 (self.conv2):
                  
                  Input 채널 수: 6
                  Output 채널 수: 16
                  커널 크기: 5x5
                  Bias 항목: 각 output 채널마다 하나
                  파라미터 수 = (입력 채널 수 * 커널 높이 * 커널 너비 + 1) * 출력 채널 수 = (6 * 5 * 5 + 1) * 16 = 2416
                  
                Fully Connected Layer 1 (self.fc1):                  
                  입력 크기: 1655
                  출력 크기: 120
                  Bias 항목: 각 출력 노드마다 하나
                  파라미터 수 = (입력 크기 + 1) * 출력 크기 = (1655 + 1) * 120 = 48120
                  
                Fully Connected Layer 2 (self.fc2):
                  입력 크기: 120
                  출력 크기: 84
                  Bias 항목: 각 출력 노드마다 하나
                  파라미터 수 = (입력 크기 + 1) * 출력 크기 = (120 + 1) * 84 = 10164
                
      
                Fully Connected Layer 3 (self.fc3):
                  입력 크기: 84
                  출력 크기: 10 (최종 클래스 수)
                  Bias 항목: 각 출력 노드마다 하나
                  파라미터 수 = (입력 크기 + 1) * 출력 크기 = (84 + 1) * 10 = 850

                총합:  156 + 2,416 + 48,120 + 10,164 + 850 = 61,706개,코드로 확인해본 결과도 동일함



              둘다 모델형태가 비슷해서 성능이 비슷할거라고 생각되고 실제로 비슷하게 나왔다.

train 및 test (main.py):
            동등한 비교 위해 하이퍼 파라메터는 두 모델 다 같은 파라메터를 사용 
              사용한 하이퍼 파라메터 
              batch_size = 64
              learning_rate = 0.01
              momentum = 0.9
              num_epochs = 10
              weight_decay=0.001 -> L2 regulazation을 위해 

              
              나머지 plot들은 파이썬 plot을 이용해서 epoch별로 loss 와 accuuracy 구해서 그림으로 저장하였다.




 #Lenet plot
 
![LeNet-5 Training and Testing Stats](https://github.com/hansanghooon/24510115_assign1/assets/132417290/9d57e439-bf3e-46ef-8a17-597d446e8b5f)



 #Custom MLP plot 


 ![Custom Training and Testing Stats](https://github.com/hansanghooon/24510115_assign1/assets/132417290/390c0e6f-dde0-44d8-a908-0b266f37b4a1)


