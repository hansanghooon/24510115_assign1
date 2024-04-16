# 24510115_assign1

데이터 처리 과정(dataset.py) : 경로를 받아 올 때 train이 True면 하위경로를 train, 아니면 하위경로에 test를 붙여서 파일 구분  
                  모든 파일은 .png이므로 png로 끝나는 애들의 이름을 전부 리스트에 담은 뒤에 Image 함수를 통해서 gray-scaleing을 해주고 
                  무슨 숫자에 대한 이미지인가를 파일 이름을 parsing해서 label 이라는 변수에 담아서 차후 분석을 진행함
                  transforms을 통해서 평균0.1307 표준편차 0.3081 를 이용해 normalize를 진행
                  main에는 제대로 된건지 확인 위해서 shape를 가져오고, label 값을 확인하는 메서드 작성해놓음



모델 (model.py):Lenet5 모델에  regurlaztion 은 weight decay와(weight decay는 main에서 적용) dropout 이용 , 
                data agumentation은 이용하지 않음, 적용하지 않은 이유는 적용하지 않아도 성능이 준수해서
                적용하지 않음. Lenet5 input이 32x32 라고 해서 28x28 이미지 사이즈를 32x32로 바꿔주기 위해  초기 convlayer에 padding 2 해줌,
                CustomMLP는 기본적으로 위에서 만든 Lenet5 모델을 그대로 이용함, 파라메터 수를 아예 같게 해주려다 보니 다른 모델의 형태를 바꿔주는대 제약이 많고, 이미 Lenet5 성능 
                을 테스트 해보니 성능이 준수해서 같은 모델에 conv layer 에 stride=2, padding=2 로설정하여 모델을 설정함  
                

                







