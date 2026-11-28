---
title: Refusals reference
description: Every refusal and error a device tool call can return - the code, the Rust variant and its fields, when it happens and what was written.
---

# Refusals reference

A device tool call ends in one of two ways: a result, or an error result. This
page lists every error result the Rust crates produce, with the exact code, so
a program can branch on it and a person can look it up.

## The shape of a result

A successful read:

```json
{"register":"DEADTIME","raw":"0x0000000c","fields":{"DEADTIME":1.2},"units":{"DEADTIME":"us"}}
```

| Key | Meaning |
| --- | --- |
| `register` | The register's name |
| `raw` | The register word as read, in hexadecimal at the register's width |
| `fields` | Every readable field: a number in physical units, a count, `true`/`false`, or an enum name |
| `units` | The unit of each field that has one; absent when none has |

A successful write:

```json
{"ok":true,"register":"DEADTIME","written":"0x0000000c","applied":{"DEADTIME":1.2},"rounded":{"DEADTIME":{"requested":1.23,"applied":1.2}},"units":{"DEADTIME":"us"}}
```

| Key | Meaning |
| --- | --- |
| `ok` | `true` |
| `register` | The register's name |
| `written` | The whole word that went to the bus |
| `applied` | The value of each field the call named, as the part now holds it |
| `rounded` | Only when a physical value was rounded to the field's resolution: what was requested and what was applied, per field |
| `units` | The unit of each named field that has one |

An error result always has the same three keys:

```json
{"ok":false,"error":"out_of_range","message":"DEADTIME = 0.2 is outside its limits 0.5 to 25.5 us. Nothing was written."}
```

| Key | Meaning |
| --- | --- |
| `ok` | `false` |
| `error` | A stable code, listed below |
| `message` | A full sentence for the model, with the field as `REGISTER.FIELD` (just `REGISTER` when the two names are the same) and limits in the part's units |

In Rust the error is a `ToolError` (`webagents::regspec::ToolError`), and
`ToolError::write_json` renders it as above. The envelope's own refusals are
the `ToolError::Refused(Refusal)` variant. On the gateway an error result is
streamed to the chat as a `tool_result` with status `error`, and given to the
model as the tool's answer.

## Refusals of the envelope

These mean the call was understood and the part's limits said no. Nothing was
written; the register keeps its value.

| `error` | Rust | When | Fields of the variant |
| --- | --- | --- | --- |
| `out_of_range` | `Refusal::OutOfRange` | The value is below the field's minimum or above its maximum, or too large to be any field's count | `field`: the field; `requested`: the value as asked, either the argument's own text or a count |
| `not_a_whole_count` | `Refusal::NotAWholeCount` | A field that counts whole steps (no scale) was given a fraction | `field`; `requested`: the number as written |
| `bad_number` | `Refusal::BadNumber` | A number with more than 18 significant digits, or one too small to represent | `field` |
| `bad_value` | `Refusal::BadEnum` | A mode field was given a name, or a number, that is not one of its named values | `field` |
| `wrong_type` | `Refusal::WrongType` | The JSON type does not fit the field: `true`/`false` for a number, a number other than 0 or 1 for a flag | `field` |
| `read_only` | `Refusal::ReadOnly` | The arguments name a field that cannot be written | `field` |
| `needs_confirmation` | `Refusal::NeedsConfirmation` | The field is marked `confirm` and the host did not state that a person confirmed this call | `field` |
| `rate_limited` | `Refusal::RateLimited` | The field was written less than its `min_interval_ms` ago | `field`; `wait_ms`: milliseconds until a write is accepted |
| `rule` | `Refusal::Rule` | A rule did not hold for the values as they would be after the write | `owner`: the field the rule is declared on; `rule`: its index in that field's rules; `lhs`, `rhs`: the two compared values, as counts |
| `must_name_all_fields` | `Refusal::MustNameAll` | The register's read has a side effect, so it is not read before writing, and the call left a writable field unnamed | `reg`: the register's index |

A field is identified by position: `FieldRef { reg, field }`, the index of the
register in the device and of the field in the register.

### Messages

The message is built from the spec, so the names and numbers below are
examples. Most come from the PWM driver spec used throughout these pages;
`STATE.BUSY`, `POS` and `FIFO_CTRL` stand for fields and registers of other
parts.

| `error` | Message |
| --- | --- |
| `out_of_range` | `DEADTIME = 0.2 is outside its limits 0.5 to 25.5 us. Nothing was written.` |
| `not_a_whole_count` | `CTRL.POLE_PAIRS counts whole steps; 2.5 is not one. Nothing was written.` |
| `bad_number` | `DEADTIME was given a number with more precision than any register can use. Nothing was written.` |
| `bad_value` | `CTRL.MODE takes one of: six_step sinusoidal foc. Nothing was written.` |
| `wrong_type` | `DEADTIME was given a value of the wrong type. Nothing was written.` |
| `read_only` | `STATE.BUSY is read only. Nothing was written.` |
| `needs_confirmation` | `SENSE.TRIP needs a person's confirmation for this write. Ask them, then the host retries with their confirmation. Nothing was written.` |
| `rate_limited` | `POS was written too recently; wait 600 ms. Nothing was written.` |
| `rule` | `the on-time must not exceed the PWM period (60 us is not <= 50 us). Nothing was written.` |
| `must_name_all_fields` | `FIFO_CTRL is not read before writing (its read has a side effect), so name every writable field. Nothing was written.` |

A `rule` message is the rule's own `why` from the spec, followed by the two
values that were compared, each in the units of the field it belongs to (an
enum name or `true`/`false` for a mode or a flag).

### Notes on single codes

- **`out_of_range`** quotes the limits the spec gives, which may be narrower
  than what the bits can hold. For a field with a negative scale the lower
  physical limit is the higher count; the message is always in physical order.
- **`bad_value`** applies to numbers too: a mode field accepts a number only if
  it is one of the named values, so a reserved encoding cannot be written by
  number. A string given to a field that has no named values is reported with
  this code as well, as `FAN.LEVEL takes a number, not a name. Nothing was written.`
- **`read_only`** needs a register that has a `set` tool. A register with no
  writable field has none, so a write to it is `unknown_tool`.
- **`wrong_type`** is not returned for `null`. An explicit `null` means "leave
  this field as it is", the same as not naming it.
- **`needs_confirmation`** cannot be satisfied by the model. The gateway runs
  every call unconfirmed; see [Trust and local limits](./trust-and-limits.md#confirmation).
- **`rule`** can name a field the call did not write. A rule declared on `DUTY`
  is checked on a `PERIOD` write as well, and the refusal then reports `DUTY`'s
  rule.
- **Rounding is not a refusal.** A physical value between two steps of a scaled
  field is applied at the nearest step and reported under `rounded`.

## Errors in the call

These mean the call could not be understood. Nothing was written.

| `error` | Rust | When | Message |
| --- | --- | --- | --- |
| `unknown_tool` | `ToolError::UnknownTool` | The name is not `<device>_read_<register>` or `<device>_set_<register>` for a register of this device that can be read, or written, that way | `No such tool on this device.` |
| `bad_arguments` | `ToolError::Json(JsonError)` | The arguments are not a JSON object (`JsonError::Syntax`), or a value is an object or an array (`JsonError::Nested`). Both carry the byte offset | `invalid JSON at byte 12.` or `nested value at byte 6: arguments must be a flat object.` |
| `unknown_argument` | `ToolError::UnknownArg(name)` | A key that is not a field of the register. A `read` tool takes no arguments, so any key is unknown there | `speed is not a field of this register.` |
| `duplicate_argument` | `ToolError::DuplicateArg(name)` | The same field named twice in one call | `deadtime is named twice.` |
| `no_arguments` | `ToolError::NoArgs` | A `set` call that names no field, or only fields set to `null` | `Name at least one field to write.` |

Field and tool names are matched without regard to case.

On the gateway, a tool name that belongs to none of its devices is answered by
the gateway itself with the same code and a different sentence:

```json
{"ok":false,"error":"unknown_tool","message":"No device has a tool named pwm9_read_status."}
```

## Failures

These are not refusals: something went wrong while carrying out a call that
was allowed.

| `error` | Rust | When | What was written |
| --- | --- | --- | --- |
| `bus_error` | `ToolError::Bus` | The transport failed: a timeout, a NACK (the part not acknowledging on I2C), a closed port, an answer that is not the protocol, or the part itself refusing the access (`E ...` on a serial line, status `0x80` on CAN) | Unknown. The message says to read the register back before retrying |
| `output_too_small` | `ToolError::OutputTooSmall` | The caller's output buffer was too small for the result | A write, if the call was one, was done. The message says so |
| `gateway_error` | (the gateway's tool loop) | The task that ran the call ended abnormally | Unknown |

```json
{"ok":false,"error":"bus_error","message":"The bus transfer failed; the register's state is unknown. Read it back before retrying."}
```

`bus_error` is also what a model sees when a part that enforces its own limits
refuses a raw write the gateway let through, which only happens when the two
were built from different specs.

`output_too_small` does not occur on the gateway, which gives every result an
8 KB buffer. In firmware, size the buffer for the largest register: a few
hundred bytes is typical.

## Errors that end a turn

A tool error is an answer the model reads and reacts to; the turn goes on. A
turn itself ends with a `response.error` frame instead of `response.done` in
these cases:

| `code` | From | When |
| --- | --- | --- |
| `model_error` | Gateway and firmware | The model call failed: the turn carried no payment token, the model socket could not be opened or went silent, the platform reported an error, or (accelerator backend) the link failed |
| `device_limit` | Firmware | The turn does not fit the device's fixed buffers: the conversation is too long, there are too many tool calls in one answer, or the tool results are too large |

When the model keeps calling tools without answering, the gateway stops after
`max_rounds` model calls (8 by default) and ends the turn normally with the
text `Stopped after 8 rounds of tool calls without a final answer.`

## Errors in a spec

A spec that does not validate is not a tool error: it never becomes a device.
`robutler-device regspec check` prints every problem at once, each with the
register and field it is in:

```text
2 problem(s) in the register spec:
  registers[0] RATE: width 12 must be 8, 16 or 32
  registers[0] RATE.RATE: a scaled field needs a unit
```

The schema id is `webagents.regspec/1`. A file that carries the older id
`reda.regspec/1` still loads, with a note asking for the file to be updated,
until the next minor release.
