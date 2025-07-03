import random
import math
from tkinter import *
import copy
from time import sleep
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class Package:
    def __init__(self, package_id, x, y, weight, priority):
        self.package_id = package_id
        self.destination = (x, y)
        self.weight = weight
        self.priority = priority

    def get_distance_from_shop(self):
        return math.sqrt(self.destination[0] ** 2 + self.destination[1] ** 2)
    

    def __repr__(self):
        return (f"Package(ID={self.package_id}, De0st={self.destination}, "
                f"Weight={self.weight}kg, Priority={self.priority})")


class Car:
    num_of_cars = 0

    def __init__(self, car_id, capacity):
        self.car_id = car_id
        self.capacity = capacity
        self.remaining_capacity = capacity
        self.assigned_packages = []
        Car.num_of_cars += 1

    def assign_package(self, package):
        if self.remaining_capacity >= package.weight:
            self.assigned_packages.append(package)
            self.remaining_capacity -= package.weight

    def get_cost_of_route(self):
        if not self.assigned_packages:
            return 0  
        
        total_distance = 0.0
        prev_x, prev_y = 0, 0 
        
        for package in self.assigned_packages:
            dx = package.destination[0] - prev_x
            dy = package.destination[1] - prev_y
            total_distance += math.sqrt(dx**2 + dy**2)
            prev_x, prev_y = package.destination  
        
        return_distance = math.sqrt(prev_x**2 + prev_y**2)
        total_distance += return_distance
        
        return total_distance

    def __repr__(self):
        return (f"Car(ID={self.car_id}, Capacity={self.capacity}kg, "
                f"Remaining={self.remaining_capacity}kg, "
                f"Packages={[len(self.assigned_packages)]})")


def show_cars_packages(root, cars):
    for widget in root.winfo_children():
        if isinstance(widget, Label):
            widget.destroy()

    row = 0
    total_cost = 0
    for car in cars:
        car_label = Label(root, text=f"Car {car.car_id}:", font=("Arial", 10, "bold"), fg="white", bg="dark green")
        car_label.grid(row=row, column=0, sticky="w", pady=2)
        row += 1
        for package in car.assigned_packages:
            package_label = Label(root, text=f"  {package}", font=("Arial", 9), fg="white", bg="dark green")
            package_label.grid(row=row, column=0, sticky="w")
            row += 1
        total_cost += car.get_cost_of_route()

    cost_label = Label(root, text=f"Total route cost after SA: {total_cost:.2f}", font=("Arial", 10, "bold"), fg="white", bg="dark green")
    cost_label.grid(row=row, column=0, sticky="w", pady=5)


def choosing_cars_to_compare(cars):
    if len(cars) < 2:
        return -1

    random.shuffle(cars)
    car1 = cars[0]
    car2 = cars[1]
    original_car1_packages = copy.deepcopy(car1.assigned_packages)
    original_car2_packages = copy.deepcopy(car2.assigned_packages)
    if len(car1.assigned_packages) < 1:
        return -1
    elif len(car2.assigned_packages) < 1:
        return -1
    
    random.shuffle(car1.assigned_packages)
    random.shuffle(car2.assigned_packages)
    package1 = car1.assigned_packages[0]
    package2 = car2.assigned_packages[0]

    return package1, package2, original_car1_packages , original_car2_packages

def simulated_annealing(packages, cars):

    num_of_iteration = 100
    temp = 1000
    stop_temp = 1
    cooling_rate = 0.91

    def angle_between_two_packages(package1, package2):
        angle = 30
        x1, y1 = package1.destination
        x2, y2 = package2.destination

        angle1 = math.degrees(math.atan2(y1, x1))  
        angle2 = math.degrees(math.atan2(y2, x2))  

        return abs(angle1 - angle2) <= angle

    while temp > stop_temp:
        for _ in range(num_of_iteration):

            if len(cars) == 1:
                if len(cars[0].assigned_packages) < 2:
                    continue

                car = cars[0]
                original_packages = copy.deepcopy(car.assigned_packages)
                package1, package2 = random.sample(car.assigned_packages, 2)

                idx1 = car.assigned_packages.index(package1)
                idx2 = car.assigned_packages.index(package2)

                same_road = angle_between_two_packages(package1, package2)
                current_cost = car.get_cost_of_route()

                car.assigned_packages[idx1], car.assigned_packages[idx2] = package2, package1

                new_cost = car.get_cost_of_route()

                if same_road and package1.get_distance_from_shop() < package2.get_distance_from_shop():
                    car.assigned_packages = original_packages
                elif not same_road and package1.priority > package2.priority:
                    car.assigned_packages = original_packages

                diff_energy = new_cost - current_cost
                if diff_energy > 0:
                    if random.random() > math.exp(-diff_energy / temp):
                        car.assigned_packages = original_packages

            else:
                random.shuffle(cars)
                car1, car2 = cars[0], cars[1]

                if len(car1.assigned_packages) < 1 or len(car2.assigned_packages) < 1:
                    continue

                original_car1 = copy.deepcopy(car1.assigned_packages)
                original_car2 = copy.deepcopy(car2.assigned_packages)

                package1 = random.choice(car1.assigned_packages)
                package2 = random.choice(car2.assigned_packages)

                idx1 = car1.assigned_packages.index(package1)
                idx2 = car2.assigned_packages.index(package2)

                same_road = angle_between_two_packages(package1, package2)

                can_car1_accept = car1.remaining_capacity + package1.weight >= package2.weight
                can_car2_accept = car2.remaining_capacity + package2.weight >= package1.weight
                if not (can_car1_accept and can_car2_accept):
                    continue

                current_cost = sum(car.get_cost_of_route() for car in cars)

                car1.assigned_packages[idx1] = package2
                car2.assigned_packages[idx2] = package1

                car1.remaining_capacity = car1.capacity - sum(p.weight for p in car1.assigned_packages)
                car2.remaining_capacity = car2.capacity - sum(p.weight for p in car2.assigned_packages)

                new_cost = sum(car.get_cost_of_route() for car in cars)

                if same_road and package1.get_distance_from_shop() < package2.get_distance_from_shop():
                    car1.assigned_packages = original_car1
                    car2.assigned_packages = original_car2
                    car1.remaining_capacity = car1.capacity - sum(p.weight for p in car1.assigned_packages)
                    car2.remaining_capacity = car2.capacity - sum(p.weight for p in car2.assigned_packages)
                    continue
                elif not same_road and package1.priority > package2.priority:
                    car1.assigned_packages = original_car1
                    car2.assigned_packages = original_car2
                    car1.remaining_capacity = car1.capacity - sum(p.weight for p in car1.assigned_packages)
                    car2.remaining_capacity = car2.capacity - sum(p.weight for p in car2.assigned_packages)
                    continue

                diff_energy = new_cost - current_cost
                if diff_energy > 0:
                    if random.random() > math.exp(-diff_energy / temp):
                        car1.assigned_packages = original_car1
                        car2.assigned_packages = original_car2
                        car1.remaining_capacity = car1.capacity - sum(p.weight for p in car1.assigned_packages)
                        car2.remaining_capacity = car2.capacity - sum(p.weight for p in car2.assigned_packages)

        temp *= cooling_rate


def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def calculate_fitness(solution, cars):
    total_distance = 0
    penalty = 0

    for car_id, package_list in solution.items():
        car_index = int(car_id)
        car = cars[car_index]
        current_weight = 0
        route_distance = 0

        current_x, current_y = 0, 0

        for package in package_list:
            current_weight += package.weight
            package.x, package.y = package.destination
            route_distance += calculate_distance(current_x, current_y, package.x, package.y)
            current_x, current_y = package.x, package.y

        route_distance += calculate_distance(current_x, current_y, 0, 0)
        total_distance += route_distance

        if current_weight > car.capacity:
            penalty += 1000 * (current_weight - car.capacity)

    return total_distance + penalty

def generate_solutions(cars, packages, population_size):
    solutions = []

    for _ in range(population_size):
        new_solution = {}

        if len(cars) == 1:
            assigned_copy = packages[:]
            random.shuffle(assigned_copy)
            new_solution["0"] = assigned_copy
        else:
            car_solution = {str(i): [] for i in range(len(cars))}
            shuffled_packages = packages[:]
            random.shuffle(shuffled_packages)

            for pkg in shuffled_packages:
                chosen_car = random.choice(list(car_solution.keys()))
                car_solution[chosen_car].append(pkg)

            new_solution = car_solution

        fitness = calculate_fitness(new_solution, cars)
        solutions.append({
            "routes": new_solution,
            "fitness": fitness
        })

    return solutions

def crossover(routes1, routes2):
    child_routes = {}
    for car_id in routes1:
        if random.random() < 0.5:
            child_routes[car_id] = routes1[car_id][:]
        else:
            child_routes[car_id] = routes2[car_id][:]
    return child_routes

def mutate(routes, mutation_rate, cars):
    if len(routes) < 2:
        return routes

    if random.random() < mutation_rate:
        car_ids = list(routes.keys())
        car1, car2 = random.sample(car_ids, 2)

        if routes[car1]:
            pkg = random.choice(routes[car1])
            cap2_used = sum(p.weight for p in routes[car2])
            cap2_total = cars[int(car2)].capacity

            if cap2_used + pkg.weight <= cap2_total:
                routes[car1].remove(pkg)
                routes[car2].append(pkg)

    return routes

def get_solution_fitness(solution):
    return solution["fitness"]

def get_best_solution(solutions):
    best = solutions[0]
    for sol in solutions[1:]:
        if sol["fitness"] < best["fitness"]:
            best = sol
    return best

def genetic_algorithm(packages, cars):
    population_size = 75
    num_of_generations = 500
    mutation_rate = 0.05

    if not packages or not cars:
        return 0

    population = generate_solutions(cars, packages, population_size)

    for generation in range(num_of_generations):
        for solution in population:
            solution["fitness"] = calculate_fitness(solution["routes"], cars)

        population.sort(key=get_solution_fitness)

        best_solutions = population[:population_size // 2]
        new_population = best_solutions.copy()

        while len(new_population) < population_size:
            parent1, parent2 = random.sample(best_solutions, 2)
            child_routes = crossover(parent1["routes"], parent2["routes"])
            mutated_child_routes = mutate(child_routes, mutation_rate, cars)
            fitness = calculate_fitness(mutated_child_routes, cars)
            new_population.append({
                "routes": mutated_child_routes,
                "fitness": fitness
            })

        population = new_population

    best_solution = get_best_solution(population)
    return best_solution


def shop_specification(root):
    package_id = 1
    cars = []
    packages = []
    capacity_entries = []

    car_frame = Frame(root, bg="dark green") 
    car_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

    num_vech_label = Label(car_frame, text="Please enter the number of the cars", font=("Arial", 12), bg="dark green", fg="white")
    num_vech_label.grid(row=0, column=0, padx=5, pady=5)

    num_vehicles_field = Entry(car_frame, width=20)
    num_vehicles_field.grid(row=1, column=0, padx=5)

    def get_num_vehicles():
        num_vehicles = int(num_vehicles_field.get())
        cap_label = Label(car_frame, text="Capacity of each car in kg:", bg="dark green", fg="white", font=("Arial", 10, "bold"))
        cap_label.grid(row=2, column=0, pady=5)

        for i in range(num_vehicles):
            cap_field = Entry(car_frame, width=5)
            cap_field.grid(row=3 + i, column=0, padx=5)
            capacity_entries.append(cap_field)

        def saving_capacity():
            cap_label2 = Label(car_frame, text="Saving the Capacities of the cars", font=("Arial", 10, "bold"), fg="white", bg="dark green")
            cap_label2.grid(row=num_vehicles + 4, column=0, pady=3)
            cars.clear()
            for i in range(len(capacity_entries)):
                entry = capacity_entries[i]
                cap = int(entry.get())
                cars.append(Car(car_id=i + 1, capacity=cap))


        cap_button = Button(car_frame, text="Save", command=saving_capacity)
        cap_button.grid(row=num_vehicles + 3, column=0, pady=5)

    num_vehicles_button = Button(car_frame, text="Submit", command=get_num_vehicles)
    num_vehicles_button.grid(row=1, column=1, padx=5)

    package_frame = Frame(root, bg="dark green")
    package_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nw")

    Label(package_frame, text="X Coordinate 1 to 100", font=("Arial", 10, "bold"), fg="white", bg="dark green").grid(row=0, column=0, pady=3)
    Label(package_frame, text="Y Coordinate 1 to 100", font=("Arial", 10, "bold"), fg="white", bg="dark green").grid(row=1, column=0, pady=3)
    Label(package_frame, text="Weight in (kg)", font=("Arial", 10, "bold"), fg="white", bg="dark green").grid(row=2, column=0, pady=3)
    Label(package_frame, text="Priority 1 to 5 (5 is the lowest)", font=("Arial", 10, "bold"), fg="white", bg="dark green").grid(row=3, column=0, pady=3)

    x_field = Entry(package_frame, width=10)
    y_field = Entry(package_frame, width=10)
    weight_field = Entry(package_frame, width=10)
    priority_field = Entry(package_frame, width=10)

    x_field.grid(row=0, column=1, padx=5)
    y_field.grid(row=1, column=1, padx=5)
    weight_field.grid(row=2, column=1, padx=5)
    priority_field.grid(row=3, column=1, padx=5)

    def get_package_fields():
        nonlocal package_id
        try:
            x = float(x_field.get())
            y = float(y_field.get())
            weight = float(weight_field.get())
            priority = int(priority_field.get())
        except ValueError:
            return

        if x > 100 or y > 100 or priority > 5 or priority < 1 or weight <= 0:
            return

        def get_remaining_capacity(car):
            return car.remaining_capacity

        def get_best_car_by_capacity(car_list):
            if not car_list:
                return None
            best_car = max(car_list, key=get_remaining_capacity)
            return best_car.car_id

        def Init_distribute_packages(cars, weight):
            eligible_cars = [car for car in cars if car.remaining_capacity >= weight]
            if eligible_cars:
                return get_best_car_by_capacity(eligible_cars)
            return None

        x_field.delete(0, END)
        y_field.delete(0, END)
        weight_field.delete(0, END)
        priority_field.delete(0, END)

        assigned = Init_distribute_packages(cars, weight)
        if assigned == 0:
            not_assigned = Label(package_frame, text="The package is overweighted", font=("Arial", 10, "bold"), fg="white", bg="dark green")
            not_assigned.grid(row=6, column=0, pady=3)
            root.after(3000, not_assigned.destroy)
            return

        package = Package(package_id, x, y, weight, priority)
        packages.append(package)


        for car in cars:
            if car.car_id == assigned:
                car.assign_package(package)
                car_capacity_label = Label(package_frame, text=f"Car {car.car_id}: remaining capacity is {car.remaining_capacity}kg", font=("Arial", 10, "bold"), fg="white", bg="dark green")
                car_capacity_label.grid(row=7, column=0, pady=3)
                root.after(3000, car_capacity_label.destroy)
                break

        l = Label(package_frame, text=f"Package id {package_id}: Point ({x}, {y}), weight: {weight}kg, priority: {priority}", font=("Arial", 10, "bold"), fg="white", bg="dark green")
        l.grid(row=6, column=0, pady=3)
        package_frame.after(3000, l.destroy)

        package_id += 1

    packages_button = Button(package_frame, width=20, command=get_package_fields, text="Submit")
    packages_button.grid(row=4, column=1, padx=5, pady=5)

    return cars, packages



def show_packages_cars(root, packages, cars, top_right):

    show_package_frame = Frame(root, bg="dark green")
    show_package_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

    show_package_label = Label(show_package_frame, text="Print each Car packages:", bg="dark green", fg="white", font=("Arial", 10, "bold"))
    show_package_label.grid(row=0, column=0, pady=5)

    chart_frame = Frame(root, bg="white")
    chart_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

    def print_packages():

        for widget in show_package_frame.winfo_children():
            if isinstance(widget, Label) and widget != show_package_label:
                widget.destroy()

        i = 1
        cost =0 
        for car in cars:
            car_id_label = Label(show_package_frame, text=f"Car {car.car_id}:", bg="dark green", fg="white", font=("Arial", 10, "bold"))
            car_id_label.grid(row=i, column=0, sticky="w")
            i += 1
            for package in car.assigned_packages:
                package_label = Label(show_package_frame, text=f"  {package}", bg="dark green", fg="white", font=("Arial", 8))
                package_label.grid(row=i, column=0, sticky="w")
                i += 1
            cost += car.get_cost_of_route()
        
        cost_label = Label(show_package_frame , text=f"{cost}",bg="dark green", fg="white", font=("Arial", 8))
        cost_label.grid(row=i , column=0, sticky="w")

        fig, ax = plt.subplots(figsize=(2, 1))
        x_vals = [package.destination[0] for package in packages]
        y_vals = [package.destination[1] for package in packages]
        ids = [package.package_id for package in packages]

        ax.scatter(x_vals, y_vals, color='blue')

        for i, txt in enumerate(ids):
            ax.annotate(txt, (x_vals[i]+1, y_vals[i]+1))

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title('Packages Destinations')

        for widget in chart_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    show_package_button = Button(show_package_frame, text="Show Packages", width=20, command=print_packages)
    show_package_button.grid(row=1, column=1, padx=5, pady=5)


def run_simulated_annealing(root, cars, packages):
    result_frame = Frame(root, bg="dark green")
    result_frame.pack(fill="both", expand=True, pady=5)

    def show_and_run():
        for widget in result_frame.winfo_children():
            widget.destroy()

        simulated_annealing(packages, cars)
        show_cars_packages(result_frame, cars)

    sim_button = Button(root, text="Run SA", width=15, height=1, font=("Arial", 10), bg="light grey", command=show_and_run)
    sim_button.pack(pady=5)

def run_genetic_algorithm(root , cars , packages):
    result_frame = Frame(root, bg="dark green")
    result_frame.pack(fill="both", expand=True, pady=5)

    def show_and_run():
        for widget in result_frame.winfo_children():
            widget.destroy()

        genetic_algorithm(packages, cars)
        show_cars_packages(result_frame, cars)

    GA_button = Button(root, text="Run GA", width=15, height=1, font=("Arial", 10), bg="light grey", command=show_and_run)
    GA_button.pack(pady=5)

def main():
    root = Tk()
    root.title("Delivery Shop")
    root.state("zoomed")

    label = Label(root, text="Welcome To Delivery Shop!", fg="Blue", font=("Arial", 20, "bold"))
    label.pack(pady=10)

    outer_frame = Frame(root, bg="dark green", bd=2, relief="solid")
    outer_frame.pack(pady=10, padx=10, fill="both", expand=True)

    left_frame = Frame(outer_frame, bg="dark green")
    left_frame.grid(row=0, column=0, sticky="ns")

    separator1 = Frame(outer_frame, bg="black", width=2)
    separator1.grid(row=0, column=1, sticky="ns")

    middle_frame = Frame(outer_frame, bg="dark green")
    middle_frame.grid(row=0, column=2, sticky="ns")

    separator2 = Frame(outer_frame, bg="black", width=2)
    separator2.grid(row=0, column=3, sticky="ns")

    right_frame = Frame(outer_frame, bg="dark green")
    right_frame.grid(row=0, column=4, sticky="nsew")

    outer_frame.columnconfigure(3, weight=1) 

    top_right = Frame(right_frame, bg="dark green", bd=1, relief="ridge")
    top_right.pack(fill="both", expand=True, pady=5)

    top_label = Label(top_right, text="Simulated Annealing", font=("Arial", 14, "bold"), fg="white", bg="dark green")
    top_label.pack(fill="both" ,expand=True , pady=5)

    bottom_right = Frame(right_frame, bg="dark green", bd=1, relief="ridge")
    bottom_right.pack(fill="both", expand=True, pady=5)

    bottom_label = Label(bottom_right, text="Genetic", font=("Arial", 14, "bold"), fg="white", bg="dark green")
    bottom_label.pack(fill="both",expand=True , pady=5)
  
    cars, packages = shop_specification(left_frame)
    show_packages_cars(middle_frame, packages, cars , top_right)
    run_simulated_annealing(top_right , cars ,packages)
    run_genetic_algorithm(bottom_right , cars, packages)

    def on_reset_button_click():
        reset_application(root)

    reset_button = Button(root, text="Reset", font=("Arial", 10, "bold"), bg="red", fg="white",command=on_reset_button_click, width=20)
    reset_button.pack(pady=5)


    root.mainloop()

def reset_application(root):
    for widget in root.winfo_children():
        widget.destroy()
    main()

main()

