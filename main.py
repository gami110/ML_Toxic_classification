from toxic_comment_classifier import ToxicCommentClassifier

def main_menu(classifier):
    while True:
        print("Меню управления моделью классификации токсичности комментариев:")
        print("1 - Обучить модели")
        print("2 - Выбрать модель обучения и предсказать комментарий")
        print("3 - Выход")

        choice = input("Введите номер действия: ")

        if choice == '1':
            classifier.run()
            print("Модель обучена и сохранена.")
        elif choice == '2':
            print("Выберите тип модели для предсказания:")
            print("1 - Стандартные модели")
            print("2 - Модели после GridSearch")
            model_type_choice = input("Введите номер типа модели: ")

            if model_type_choice == '1':
                model_type = 'standard'
            elif model_type_choice == '2':
                model_type = 'grid_search'
            else:
                print("Неверный номер типа модели.")
                continue

            print("Выберите модель для предсказания:")
            if model_type == 'standard':
                model_names = list(classifier.models.keys())
            else:
                model_names = [name + "_best" for name in list(classifier.models.keys())]

            for i, model_name in enumerate(model_names, start=1):
                print(f"{i} - {model_name}")
            print(f"{len(model_names) + 1} - Все модели")

            model_choice = input("Введите название модели или номер: ")
            if model_choice.isdigit():
                model_index = int(model_choice)
                if model_index == len(model_names) + 1:
                    model_choice = "Все"
                elif 1 <= model_index <= len(model_names):
                    model_choice = model_names[model_index - 1]
                else:
                    print("Неверный номер модели.")
                    continue
            elif model_choice not in model_names:
                print("Неверное название модели.")
                continue

            comment = input("Введите комментарий: ")
            predictions = {}
            if model_choice == "Все":
                # Сначала загружаем все модели
                for model_name in model_names:
                    classifier.load_model(model_name)
                # Затем делаем предсказания для всех моделей
                for model_name in model_names:
                    prediction = classifier.predict_comment(comment, model_name=model_name)
                    predictions[model_name] = prediction
                # Выводим результаты предсказаний
                for model_name, prediction in predictions.items():
                    print(f"Результат от модели {model_name}: {'Токсичный' if prediction[0] == 1 else 'Не токсичный'}")
            else:
                classifier.load_model(model_choice)
                prediction = classifier.predict_comment(comment, model_name=model_choice)
                print(f"Результат от модели {model_choice}: {'Токсичный' if prediction[0] == 1 else 'Не токсичный'}")
        elif choice == '3':
            print("Выход из программы.")
            break
        else:
            print("Неверный ввод. Пожалуйста, попробуйте еще раз.")



if __name__ == '__main__':
    classifier = ToxicCommentClassifier()
    main_menu(classifier)