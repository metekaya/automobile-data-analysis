from termcolor import colored
import timeit

start_time = timeit.default_timer()


def stop_timer():
    elapsed = timeit.default_timer() - start_time
    print(colored('\n****************************************************************', 'blue'), )
    print(colored(f'Kodun Çalışma Süresi: {elapsed * 1000} ms', 'blue'), )
    print(colored('****************************************************************', 'blue'), )

# Import this file like --> from timer import stop_timer
# And call the function at the end of your code ---> stop_timer()