
def update_model_params(model_A, model_B, alpha):
    """
    model_A : source model(pseudo labeler)
    model_B : target model
    """
    # 모델 A는 CPU, B는 GPU에 있음
    device = next(model_A.parameters()).device  # A의 장치를 가져옴 (CPU)
    
    # 모델 A와 B의 파라미터를 반복하면서 업데이트
    for param_A, param_B in zip(model_A.parameters(), model_B.resnet.parameters()):
        # B의 파라미터를 A의 장치(CPU)로 이동
        param_B_cpu = param_B.to(device)
        
        # param_A를 (1 - alpha) * param_A + alpha * param_B로 업데이트
        param_A.data = (1 - alpha) * param_A.data + alpha * param_B_cpu.data