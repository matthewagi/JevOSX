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
