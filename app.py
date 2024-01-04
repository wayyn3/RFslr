def sign_language_recognition_with_mediapipe():
    st.write('Perform the sign language gesture in front of your webcam...')
    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        H, W, _ = cv2_img.shape

        frame_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame_rgb,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            data_aux = []
            x_ = []
            y_ = []

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

            expected_num_features = model.n_features_in_

            if len(data_aux) < expected_num_features:
                data_aux.extend([0] * (expected_num_features - len(data_aux)))
            elif len(data_aux) > expected_num_features:
                data_aux = data_aux[:expected_num_features]

            prediction = load_model().predict([np.asarray(data_aux)])

            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame_rgb, labels_dict[prediction[0]], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

            st.image(frame_rgb, channels='RGB', use_column_width=True)
