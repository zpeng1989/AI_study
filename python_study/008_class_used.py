class Student(object):
    def __init__(self, name, age):
        self.name = name
        self.age = age
    def study(self, course_name):
        print('At age at %s with study by %s'%(self.name, course_name))

    def watch_movie(self):
        if self.age < 18:
            print('%s only watch.'% self.name)
        else:
            print('%s avv' % self.name)



def main():
    stu1 = Student('aa', 38)
    stu1.study('python class')
    stu1.watch_movie()
    stu2 = Student('bb', 15)
    stu2.study('sixiang')
    stu2.watch_movie()

if __name__ == '__main__':
    main()