def is_leap_year(year):
    return year % 4 == 0 and year % 100 != 0 or year %400 == 0

def which_day(year, month, date):
    days_of_month = [[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],[31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]][is_leap_year(year)]
    total = 0
    for index in range(month - 1):
        total += days_of_month[index]
    return total + date


def main():
    print(which_day(1980, 11, 28))
    print(which_day(1981, 12, 31))
    print(which_day(1983, 4, 21))


if __name__ == '__main__':
    main()

