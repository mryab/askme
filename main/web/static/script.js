$(function () {
    var $story = $('#story'),
        $question = $('#question'),
        $answer = $('#answer'),
        $getAnswer = $('#get_answer'),
        $getStory = $('#get_story')

    getStory();

    // Activate tooltip
    $('.qa-container').find('.glyphicon-info-sign').tooltip();

    $getAnswer.on('click', function (e) {
        e.preventDefault();
        getAnswer();
    });

    $getStory.on('click', function (e) {
        e.preventDefault();
        getStory();
    });

    function getStory() {
        $.get('/get/story', function (json) {
            $story.val(json["story"]);
            $story.data('original_story', json["story"]);
            $question.val(json["question"]);
            $question.data('question_idx', json["question_idx"]);
            $question.data('suggested_question', json["question"]); // Save suggested question
            $answer.val('');
            $answer.data('correct_answer', json["correct_answer"]);
        });
    }

    function getAnswer() {
        var questionIdx = $question.data('question_idx'),
            correctAnswer = $answer.data('correct_answer'),
            suggestedQuestion = $question.data('suggested_question'),
            question = $question.val(), story = $story.val(), originalStory = $story.data('original_story');

        var userQuestion = suggestedQuestion !== question ? question : '';
        var userStory = story !== originalStory ? story : '';
        var url = '/get/answer?question_idx=' + questionIdx +
            '&user_question=' + encodeURIComponent(userQuestion) + '&user_story=' + encodeURIComponent(userStory);

        $.get(url, function (json) {
            var predAnswer = json["pred_answer"];

            var outputMessage = "Answer = " + predAnswer;

            // Show answer's feedback only if suggested question was used
            if (userQuestion === '' && userStory === '') {
                if (predAnswer === correctAnswer)
                    outputMessage += "\nCorrect!";
                else
                    outputMessage += "\nWrong. The correct answer is '" + correctAnswer + "'";
            }
            $answer.val(outputMessage);

        });
    }
});
