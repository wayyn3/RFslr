def sign_language_recognition_with_mediapipe():
    cap = cv2.VideoCapture(0)  
    image_placeholder = st.empty()

    expected_num_features = model.n_features_in_
    
    while True:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()

        flipped_frame = cv2.flip(frame, 1)

        if not ret:
            print("Error: Failed to grab a frame.")
            break

        H, W, _ = flipped_frame.shape

        frame_rgb = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    flipped_frame,  
                    hand_landmarks,  
                    mp_hands.HAND_CONNECTIONS,  
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            if len(data_aux) < expected_num_features:
                data_aux.extend([0] * (expected_num_features - len(data_aux)))
            elif len(data_aux) > expected_num_features:
                data_aux = data_aux[:expected_num_features]

            prediction = model.predict([np.asarray(data_aux)])

            print("Predicted Value:", prediction[0])

            cv2.rectangle(flipped_frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(flipped_frame, prediction[0], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

        image_placeholder.image(flipped_frame, channels='BGR', use_column_width=True)

    cap.release()
