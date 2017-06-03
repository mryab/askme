$(function () {
    var $story = $('#story'),
        $question = $('#question'),
        $answer = $('#answer'),
        $getAnswer = $('#get_answer'),
        $getStory = $('#get_story');

    getStory();

    $('.form-horizontal').find('.glyphicon-info-sign').tooltip();

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
            $question.data('suggested_question', json["question"]);
            $answer.val('');
            $answer.data('correct_answer', json["correct_answer"]);
        });
    }

    function getAnswer() {
        var correctAnswer = $answer.data('correct_answer'),
            suggestedQuestion = $question.data('suggested_question'),
            question = $question.val(), story = $story.val(), originalStory = $story.data('original_story');

        var url = '/get/answer?question=' + encodeURIComponent(question) + '&story=' + encodeURIComponent(story);

        $.get(url, function (json) {
            var predAnswer = json["pred_answer"];

            var outputMessage = "Answer = " + predAnswer;

            if (question === suggestedQuestion && story === originalStory) {
                if (predAnswer === correctAnswer)
                    outputMessage += "\nCorrect!";
                else
                    outputMessage += "\nWrong. The correct answer is '" + correctAnswer + "'";
            }
            $answer.val(outputMessage);

        });
    }
});
