class Variable:
    def __init__(self,data):
        if data is not None:    #추가
            if not isinstance(data, np.ndarray):
                raise TypeError("{}은 지원하지 않습니다.".format(type(data)))
        
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:   
            self.grad = np.ones_like(self.data)
        
        funcs = [self.creator]
        while funcs:
            f = funcs.pop() 
            x, y = f.input, f.output 
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x) 
        output = Variable(y)
        output.set_Creator(self) # 출력 변수에 창조자 설정
        self.input = input   # 입력 변수 기억
        self.output = output # 출력 저장
        return output
        
    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        return x**2


    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx
    

#y=e^x 함수 구현
class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)


#수치 미분
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data - eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

#합성함수   
def f(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))

def as_array(x):
    if np.isscallar(x):
        return np.array(x)
    return x

#실행코드
A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))

a = A(x)
b = B(a)
c = C(b)

#계산 그래프의 노드들을 거꾸로 거슬러 올라간다.
assert y.creator == C
assert y.creator.input == b
assert y.creator.input.creator == B
assert y.creator.input.creator.input == a
assert y.creator.input.creator.input.creator == A
assert y.creator.input.creator.input.creator.input == x


'''
#역전파에 대응하는 Variable 클래스 구현
class Variable:
    def __init__(self,data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        f = self.creator #함수 가져오기
        if f is not None:
            x = f.input  #함수의 입력 가져오기
            x.grad = f.backward(self.grad) #함수의 backward 메서드 호출
            x.backward() #하나 앞 변수의 backward 메서드 호출 (재귀)




'''
