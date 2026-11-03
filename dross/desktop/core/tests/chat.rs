use oscar_desktop::chat::{ChatWindow, Request, wrap};
use oscar_desktop::protocol::{Input, Key, Message, Mods, Packet};

fn key(w: &mut ChatWindow, k: Key) {
    w.input(Input::Key { key: k, mods: Mods::default() });
}

fn typed(w: &mut ChatWindow, text: &str) {
    for c in text.chars() {
        key(w, Key::Character(c));
    }
}

fn opened() -> ChatWindow {
    let mut w = ChatWindow::new(520, 400).unwrap();
    w.receive(Packet::new(1, 7, Message::Created { width: 520, height: 400 })).unwrap();
    w
}

#[test]
fn wraps_at_spaces_and_long_words() {
    assert_eq!(wrap("one two three", 7), ["one two", "three"]);
    assert_eq!(wrap("abcdefghij", 4), ["abcd", "efgh", "ij"]);
    assert_eq!(wrap("a\nb", 10), ["a", "b"]);
}

#[test]
fn enter_sends_once_and_waits_for_the_reply() {
    let mut w = opened();
    typed(&mut w, "hi Oscar");
    key(&mut w, Key::Enter);
    assert_eq!(w.take_requests(), [Request::Send("hi Oscar".into())]);
    assert!(w.busy);
    typed(&mut w, "again");
    key(&mut w, Key::Enter);
    assert!(w.take_requests().is_empty());
    w.reply_start();
    w.reply_text("Scram!");
    w.reply_done();
    assert!(!w.busy);
    key(&mut w, Key::Enter);
    assert_eq!(w.take_requests(), [Request::Send("again".into())]);
}

#[test]
fn a_failed_reply_gives_the_message_back() {
    let mut w = opened();
    typed(&mut w, "hello");
    key(&mut w, Key::Enter);
    w.take_requests();
    w.reply_start();
    w.reply_failed("Can't reach the server");
    assert!(!w.busy);
    key(&mut w, Key::Enter);
    assert_eq!(w.take_requests(), [Request::Send("hello".into())]);
}

#[test]
fn frames_upload_after_creation() {
    let mut w = opened();
    let mut packets = 0;
    w.frame(|_| {
        packets += 1;
        Ok(())
    })
    .unwrap();
    assert!(packets > 1);
}

#[test]
fn an_opening_message_sends_like_typing() {
    let mut w = opened();
    w.say("hello there");
    assert_eq!(w.take_requests(), [Request::Send("hello there".into())]);
    assert!(w.busy);
}
