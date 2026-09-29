from jevosx.observer.walker import TreeWalker, WalkLimits
from tests.fakes import FakeNode


def window(*children, **kwargs):
    return FakeNode("AXWindow", Title="Doc", frame=(0, 0, 800, 600), children=children, **kwargs)


def test_indexes_controls_in_reading_order_with_labels_values_and_ops():
    root = window(
        FakeNode("AXButton", Title="New", frame=(10, 10, 50, 20)),
        FakeNode("AXStaticText", Value="Name:", frame=(10, 40, 50, 20)),
        FakeNode("AXTextField", Value="Ada", settable=["AXValue"], Focused=True, frame=(70, 40, 200, 20)),
        FakeNode("AXCheckBox", Title="Remember me", Value=1, frame=(10, 70, 100, 20)),
        FakeNode("AXButton", Title="Delete", Enabled=False, frame=(10, 100, 50, 20)),
    )
    result = TreeWalker().walk([(root, None)])
    labels = [(e.index, e.role_name, e.label) for e in result.elements]
    assert labels == [
        (1, "button", "New"),
        (2, "textfield", "Name"),  # unlabeled field named by the preceding sibling text
        (3, "checkbox", "Remember me"),
        (4, "button", "Delete"),
    ]
    field = result.elements[1]
    assert field.value == "Ada" and field.focused and field.ops == ("TYPE_TEXT", "CLICK")
    assert result.elements[2].checked is True and result.elements[2].value is None
    assert result.elements[3].ops == () and "disabled" in result.elements[3].states()
    assert "Name:" in result.text


def test_prunes_offscreen_subtrees_and_skips_window_chrome():
    scroll = FakeNode(
        "AXScrollArea",
        frame=(0, 0, 800, 400),
        children=[
            FakeNode("AXLink", Title="Visible", frame=(10, 10, 80, 20)),
            FakeNode(
                "AXGroup",
                frame=(0, 900, 800, 400),
                children=[FakeNode("AXLink", Title="Hidden", frame=(10, 910, 80, 20))],
            ),
        ],
    )
    root = window(FakeNode("AXButton", Subrole="AXMinimizeButton", frame=(0, 0, 10, 10)), scroll)
    result = TreeWalker().walk([(root, None)])
    assert [e.label for e in result.elements] == ["Visible"]
    assert len(result.scroll_areas) == 1 and result.scroll_areas[0].ops == ("SCROLL_UP", "SCROLL_DOWN")


def test_rows_and_unlabeled_links_take_their_descendant_text():
    table = FakeNode(
        "AXTable",
        Description="Messages",
        VisibleRows=[
            FakeNode(
                "AXRow",
                children=[
                    FakeNode("AXCell", children=[FakeNode("AXStaticText", Value="Alice")]),
                    FakeNode("AXCell", children=[FakeNode("AXTextField", Value="Lunch on Friday?")]),
                ],
            )
        ],
    )
    link = FakeNode("AXLink", children=[FakeNode("AXStaticText", Value="Read more")])
    result = TreeWalker().walk([(window(table, link), None)])
    row, anchor = result.elements
    assert row.kind == "row" and row.label == "Alice · Lunch on Friday?"
    assert row.container == 'table "Messages"'
    assert anchor.label == "Read more"
    assert "Alice" not in result.text  # owned text is not duplicated into page text


def test_generic_groups_are_indexed_only_when_pressable():
    root = window(
        FakeNode("AXGroup", Description="Card A", actions=["AXPress"]),
        FakeNode("AXGroup", Description="Decoration", actions=["AXShowMenu"]),
    )
    result = TreeWalker().walk([(root, None)])
    assert [e.label for e in result.elements] == ["Card A"]


def test_read_only_text_fields_and_stale_nodes_are_not_controls():
    root = window(
        FakeNode("AXTextField", Value="static label"),
        FakeNode("AXButton", Title="Gone", stale=True),
        FakeNode("AXTextField", Subrole="AXSecureTextField", Value="hunter2"),
    )
    result = TreeWalker().walk([(root, None)])
    assert len(result.elements) == 1
    secure = result.elements[0]
    assert secure.secure and secure.value is None and secure.role_name == "password field"
    assert "static label" in result.text and "hunter2" not in result.text


def test_budgets_truncate_and_report():
    buttons = [FakeNode("AXButton", Title=f"B{i}") for i in range(50)]
    result = TreeWalker(WalkLimits(max_elements=10)).walk([(window(*buttons), None)])
    assert len(result.elements) == 10 and result.truncated
    ticks = iter(range(1000))
    result = TreeWalker(WalkLimits(time_budget_s=5), clock=lambda: next(ticks)).walk([(window(*buttons), None)])
    assert result.truncated and "time budget reached" in result.notes


def test_open_menu_root_gets_its_own_container():
    menu = FakeNode("AXMenu", children=[FakeNode("AXMenuItem", Title="Copy Link")])
    result = TreeWalker().walk([(window(), None), (menu, "open menu")])
    assert result.elements[0].label == "Copy Link"
    assert result.elements[0].container == "menu"


def test_focused_or_search_fields_are_typeable_even_without_settable_value():
    root = window(
        FakeNode("AXTextField", Description="Address and search bar", Focused=True),  # Chrome omnibox
        FakeNode("AXTextField", Subrole="AXSearchField", Description="Search"),
        FakeNode("AXTextField", Value="just a label"),
    )
    result = TreeWalker().walk([(root, None)])
    assert [(e.label, e.ops) for e in result.elements] == [
        ("Address and search bar", ("TYPE_TEXT", "CLICK")),
        ("Search", ("TYPE_TEXT", "CLICK")),
    ]
    assert "just a label" in result.text


def test_unknown_role_containers_are_walked_not_skipped():
    toolbar = FakeNode(
        "AXUnknown", children=[FakeNode("AXTextField", Description="Address and search bar", Focused=True)]
    )
    result = TreeWalker().walk([(window(FakeNode("AXUnknown", children=[toolbar])), None)])
    assert [e.label for e in result.elements] == ["Address and search bar"]


def test_focusable_text_field_is_typeable_but_plain_label_is_not():
    root = window(
        FakeNode("AXTextField", Description="Address and search bar", settable=["AXFocused"]),
        FakeNode("AXTextField", Value="a label"),
    )
    result = TreeWalker().walk([(root, None)])
    assert [(e.label, e.ops) for e in result.elements] == [("Address and search bar", ("TYPE_TEXT", "CLICK"))]


def test_walk_reports_why_subtrees_were_skipped():
    root = window(
        FakeNode("AXButton", Title="Gone", stale=True),
        FakeNode("AXGroup", Hidden=True, children=[FakeNode("AXButton", Title="Inside hidden")]),
        FakeNode(
            "AXScrollArea",
            frame=(0, 0, 800, 400),
            children=[FakeNode("AXButton", Title="Scrolled away", frame=(0, 5000, 50, 20))],
        ),
        FakeNode("AXScrollBar"),
        FakeNode("AXButton", Title="Visible", frame=(10, 10, 50, 20)),
    )
    result = TreeWalker().walk([(root, None)])
    assert [e.label for e in result.elements] == ["Visible"]
    assert result.skipped == {"vanished": 1, "hidden": 1, "offscreen": 1, "skipped_role": 1}


def test_window_frame_does_not_hide_content_with_inconsistent_geometry():
    # Chrome, frontmost, new window: the window reports one frame, its content another.
    root = FakeNode(
        "AXWindow",
        Title="New Tab - Google Chrome",
        frame=(0, 0, 1470, 801),
        children=[
            FakeNode(
                "AXGroup",
                frame=(3000, 900, 1470, 801),
                children=[FakeNode("AXTextField", Description="Address and search bar", Focused=True,
                                   frame=(3100, 950, 880, 24))],
            )
        ],
    )  # fmt: skip
    result = TreeWalker().walk([(root, None)])
    assert [e.label for e in result.elements] == ["Address and search bar"]


def test_unclipped_rewalk_when_everything_looked_offscreen():
    web = FakeNode(
        "AXWebArea",
        Title="Page",
        frame=(0, 0, 800, 600),
        children=[FakeNode("AXGroup", frame=(0, 5000, 800, 600), children=[FakeNode("AXLink", Title="Only link",
                                                                                    frame=(0, 5000, 80, 20))])],
    )  # fmt: skip
    result = TreeWalker().walk([(window(web), None)])
    assert [e.label for e in result.elements] == ["Only link"]
    assert any("unclipped re-walk" in note for note in result.notes)


def test_controls_listed_twice_by_the_tree_are_offered_once():
    def toolbar():  # Chrome lists its toolbar under two parents: the same controls, the same place on screen
        return FakeNode(
            "AXToolbar",
            frame=(0, 0, 800, 40),
            children=[
                FakeNode("AXButton", Title="Reload", frame=(80, 5, 30, 30)),
                FakeNode(
                    "AXTextField", Description="Address and search bar", settable=["AXValue"], frame=(120, 8, 600, 24)
                ),
            ],
        )

    page = FakeNode("AXButton", Title="Reload", frame=(300, 300, 80, 30))  # same label elsewhere: a different control
    root = window(FakeNode("AXGroup", children=[toolbar()]), FakeNode("AXGroup", children=[toolbar()]), page)
    result = TreeWalker().walk([(root, None)])
    labels = [(e.index, e.label) for e in result.elements]
    assert labels == [(1, "Reload"), (2, "Address and search bar"), (3, "Reload")]
